"""귀속·교란 실험 오케스트레이션 (논문 4장 표 3~5, 그림 5~6).

문서 × 필드 × 귀속기법 × 집계수준 × 마스킹조건 을 훑으며 지표를 낸다.
결과는 JSONL로 한 줄씩 즉시 기록하므로 중간에 죽어도 앞부분이 남고,
같은 --out 으로 다시 돌리면 남은 것만 이어서 계산한다.

사용 예
  python kodx_eval.py --model _exp/runs/layoutxlm_seed1/last \
      --docs _pilot/docs.jsonl --images _pilot/images \
      --annotations _pilot/annotations_kim.jsonl \
      --out _exp/results/seed1.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kodx_data import FIELDS, LABELS, LABEL2ID, build_examples      # noqa: E402
from kodx_units import Tagger, aggregate, subword_char_spans, units_to_word_idx
from kodx_metrics import aopc, comp_suff, naopc, plausibility
import kodx_attrib as A                                             # noqa: E402

LEVELS = ["subword", "morpheme", "eojeol"]
MASKS = ["all", "content", "functional"]        # 형태소 수준에서만 의미가 있다
METHODS = ["ig", "rollout", "random"]
KS = [1, 3, 5, 10]


def token_idx_for_words(spans, word_idx: set) -> list[int]:
    return [t for t, s in enumerate(spans) if s and s[0] in word_idx]


def topk_excluding(units: list, scores: list, k: int, protect: set) -> list[int]:
    """상위 k 어절. **보호 집합은 순위에서 아예 빼고 다음 순위로 k를 채운다.**

    ⚠ 왜 빼는가 — KIE 는 BIO 시퀀스 라벨링이라 예측 대상이 「그 토큰의 라벨」이다.
    정답 값 어절을 마스킹하면 라벨을 못 맞히는 것이 **정의상 당연**하므로, 그대로 재면
    충실도가 타당성의 복사본이 된다. 실측(구 정의): IG 의 COMP@10 0.881 vs
    recall@10 0.944, 3,498건 중 82.9%에서 정답이 통째로 마스킹됐다.

    ERASER 의 COMP 는 문서 분류(예측이 특정 토큰과 분리됨)를 전제한 지표라 KIE 에
    그대로 옮기면 이 순환이 생긴다.
    """
    ranked = sorted(range(len(scores)), key=lambda i: -scores[i])
    out, seen = [], set()
    for i in ranked:
        wi = units[i].get("word_idx")
        if wi is None or wi in seen or wi in protect:
            continue
        seen.add(wi)
        out.append(wi)
        if len(out) >= k:
            break
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="귀속·교란 실험")
    ap.add_argument("--model", required=True)
    ap.add_argument("--docs", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--annotations", default=None, help="검수 주석. 없으면 자동 라벨을 정답으로 쓴다")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ig-steps", type=int, default=32)
    # 지터 폭은 여러 개를 잰다. 파일럿 실측 자연 변이(0~1000 좌표계) 표준편차 평균이
    # 66.8 이므로 그 0.4배·1.0배·2.0배로 잡는다. 하나만 쓰면 「반응 없음」이 모델의
    # 성질인지 δ 가 작아서인지 구분되지 않는다.
    ap.add_argument("--jitter", type=int, nargs="+", default=[25, 67, 134],
                    help="레이아웃 지터 ±delta 목록 (0~1000 좌표계)")
    ap.add_argument("--save-phi", default=None,
                    help="귀속 점수를 저장할 JSONL. 있으면 다음부터 충실도만 재계산할 수 있다")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--skip-naopc", action="store_true", help="NAOPC 생략. 먼저 AOPC만 뽑을 때 쓴다")
    ap.add_argument("--beam-size", type=int, default=5,
                    help="NAOPC 상한·하한 빔 서치 크기. 논문 기본값 5")
    ap.add_argument("--naopc-max-steps", type=int, default=None,
                    help="빔 서치 단계 상한. 자르면 논문 정의에서 벗어나므로 명시할 것")
    a = ap.parse_args()

    from transformers import AutoModelForTokenClassification
    from kodx_data import load_processor
    from PIL import Image

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = load_processor(a.model, apply_ocr=False)
    model = AutoModelForTokenClassification.from_pretrained(a.model).to(dev).eval()
    tok = proc.tokenizer
    mask_id = tok.mask_token_id
    gen = torch.Generator(device="cpu").manual_seed(a.seed)

    tagger = Tagger()
    levels = LEVELS if tagger.available else ["subword", "eojeol"]
    if not tagger.available:
        print("! MeCab-ko 없음 — 형태소 수준을 건너뛴다. 논문 표 3에는 필요하다.")

    ex = build_examples(Path(a.docs), Path(a.images),
                        Path(a.annotations) if a.annotations else None)
    if a.limit:
        ex = ex[: a.limit]
    print(f"평가 문서 {len(ex)}건 / 장치 {dev} / 집계수준 {levels}")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.open(encoding="utf-8"):
            try:
                r = json.loads(line)
                done.add((r["doc_id"], r["field"], r["method"], r["level"], r["mask"]))
            except ValueError:
                continue
        print(f"이미 계산된 조합 {len(done)}개 — 건너뛴다")

    f_out = out.open("a", encoding="utf-8")
    f_phi = Path(a.save_phi).open("a", encoding="utf-8") if a.save_phi else None

    def emit(rec: dict) -> None:
        f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f_out.flush()
        os.fsync(f_out.fileno())

    for n_doc, e in enumerate(ex):
        if not e.gold_spans:
            continue
        img = Image.open(e.image_path).convert("RGB")
        enc = proc(img, e.words, boxes=e.boxes, truncation=True,
                   padding="max_length", max_length=512,
                   return_offsets_mapping=True, return_tensors="pt")
        offsets = enc.pop("offset_mapping")
        spans = subword_char_spans(tok, e.words, {"word_ids": enc.word_ids(),
                                                  "offset_mapping": offsets[0]})
        inputs = {k: v.to(dev) for k, v in enc.items()}

        with torch.no_grad():
            base_logits = model(**inputs).logits
        pred_ids = base_logits.argmax(-1)[0]

        embed_fn = model.get_input_embeddings()

        def score_fn(m, inp, inputs_embeds=None, token_idx=None, label_ids=None):
            kw = dict(inp)
            if inputs_embeds is not None:
                kw.pop("input_ids", None)
                kw["inputs_embeds"] = inputs_embeds
            lg = m(**kw).logits
            return A.field_score(lg, token_idx, label_ids)

        for field, gold in e.gold_spans.items():
            gold_widx = set(gold["word_idx"])
            if not gold_widx:
                continue
            tgt_tokens = token_idx_for_words(spans, gold_widx)
            if not tgt_tokens:
                continue
            label_ids = [int(pred_ids[t]) for t in tgt_tokens]

            def sc(inputs_embeds=None, ov=None):
                inp = dict(inputs) if ov is None else {**inputs, **ov}
                return score_fn(model, inp, inputs_embeds, tgt_tokens, label_ids)

            with torch.no_grad():
                p_full = float(sc())

            # 교란 단위는 **어절**이다. 토큰 단위로 하면 N이 512까지 커져
            # 빔 서치 비용이 O(B·N²)로 폭발한다. 어절이면 N≈122로 4배 줄고,
            # 타당성·충실도 계산도 이미 어절 단위라 일관된다.
            def eval_masked(word_sel):
                tk = token_idx_for_words(spans, set(word_sel))
                if not tk:
                    return p_full
                with torch.no_grad():
                    return float(sc(ov={"input_ids": A.mask_tokens(
                        inputs["input_ids"], tk, mask_id)}))

            # --- AOPC 상한·하한은 (문서, 필드)에만 의존한다 ---
            # 그 모델이 그 입력에서 낼 수 있는 한계값이므로 귀속 기법·집계 수준·
            # 마스킹 조건과 무관하다. 안쪽 루프에서 구하면 같은 값을 조합 수(15회)
            # 만큼 다시 계산하게 된다. 여기서 한 번만 구해 재사용한다.
            lo = up = None
            if not a.skip_naopc:
                # 후보에서도 보호 집합을 뺀다 — 상한·하한이 「정답을 지우는 순서」를
                # 최선으로 고르면 NAOPC 의 분모가 순환으로 부풀어 오른다
                cand = sorted({s[0] for s in spans if s is not None} - gold_widx)
                up = A.beam_limit(eval_masked, p_full, cand, a.beam_size,
                                  "upper", a.naopc_max_steps)
                lo = A.beam_limit(eval_masked, p_full, cand, a.beam_size,
                                  "lower", a.naopc_max_steps)

            for method in METHODS:
                if method == "ig":
                    phi = A.integrated_gradients(
                        model, inputs, lambda ids: embed_fn(ids),
                        lambda m, inp, inputs_embeds: score_fn(
                            m, inp, inputs_embeds, tgt_tokens, label_ids),
                        steps=a.ig_steps, baseline_token_id=mask_id).abs()
                elif method == "rollout":
                    with torch.no_grad():
                        att = model(**inputs, output_attentions=True).attentions
                    phi = A.attention_rollout(att, tgt_tokens[0])
                else:
                    phi = A.random_attribution(base_logits.shape[1], gen)

                phi = phi.detach().float().cpu().tolist()
                # 귀속 점수를 남긴다. 이것이 있으면 충실도 정의를 바꿔도 ③단계만
                # 다시 돌릴 수 있다 — 이번 순환 문제로 전체를 재실행한 이유가
                # 이 파일이 없었기 때문이다.
                if f_phi is not None:
                    f_phi.write(json.dumps(
                        {"doc_id": e.doc_id, "field": field, "method": method,
                         "phi": [round(v, 6) for v in phi]}, ensure_ascii=False) + "\n")

                for level in levels:
                    for mask in (MASKS if level == "morpheme" else ["all"]):
                        key = (e.doc_id, field, method, level, mask)
                        if key in done:
                            continue

                        scores, units = aggregate(phi, spans, e.words, level,
                                                  tagger, mask)
                        if not scores:
                            continue

                        rec = {"doc_id": e.doc_id, "field": field, "method": method,
                               "level": level, "mask": mask, "p_full": p_full,
                               "plaus": {}, "faith": {}}

                        # --- 3.3.1 타당성 ---
                        # 다른 안(案)의 어절은 오답이 아니라 제외다. 한 장에 시행문이
                        # 여러 벌 붙는 「1건 기안 다수 시행」 때문이며, 정답은 최상단 안
                        # 하나다(가이드 1-1). 파일럿의 23.0%가 해당한다.
                        for k in KS:
                            widx = units_to_word_idx(units, scores, k)
                            rec["plaus"][str(k)] = plausibility(
                                widx, sorted(gold_widx), e.raw_boxes,
                                ignore_word_idx=e.ignore_word_idx)

                        # --- 3.3.2 충실도 (텍스트 모달) ---
                        # **정답 스팬은 보호 집합이다. 텍스트 마스킹에서 절대 지우지 않는다.**
                        #   COMP@k : 보호 집합을 뺀 상위 k 를 마스킹
                        #   SUFF@k : 보호 집합 + 상위 k 만 남기고 나머지를 마스킹
                        # SUFF 에서 보호 집합을 남기지 않으면 순환이 COMP 에서 SUFF 로
                        # 옮겨갈 뿐이다 — 「top-k만 남긴다」가 곧 정답을 지우는 것이 된다.
                        comps, suffs = [], []
                        special = [t for t, s in enumerate(spans) if s is None]
                        gold_tk = token_idx_for_words(spans, gold_widx)
                        for k in KS:
                            widx = set(topk_excluding(units, scores, k, gold_widx))
                            tk = token_idx_for_words(spans, widx)
                            if not tk:
                                comps.append(None); suffs.append(None); continue
                            with torch.no_grad():
                                p_wo = float(sc(ov={"input_ids": A.mask_tokens(
                                    inputs["input_ids"], tk, mask_id)}))
                                # 보호 집합을 함께 남긴다
                                p_only = float(sc(ov={"input_ids": A.keep_only(
                                    inputs["input_ids"], sorted(set(tk) | set(gold_tk)),
                                    mask_id, special)}))
                            cs = comp_suff(p_full, p_wo, p_only)
                            comps.append(cs["comp"]); suffs.append(cs["suff"])
                        rec["faith"]["comp@k"] = comps
                        rec["faith"]["suff@k"] = suffs
                        rec["faith"]["aopc_comp"] = aopc(comps)

                        # --- 3.3.3 모달별 교란 (레이아웃 / 이미지) ---
                        # **대상은 근거 영역(정답 스팬)이다** — 원고 2.3.3 문구 그대로.
                        # 상위 귀속 단위를 대상으로 하면 모달 민감도가 귀속 기법에 따라
                        # 달라져, 「모델의 모달 의존」이 아니라 「기법의 성질」을 재게 된다.
                        #
                        # 텍스트와 달리 여기서는 보호하지 않는다. 좌표를 흔들거나 이미지를
                        # 가려도 **OCR 텍스트는 그대로 남아** 값을 읽을 수 있으므로 순환이
                        # 생기지 않는다. 사라지는 것은 위치·시각 정보뿐이다.
                        if gold_tk:
                            with torch.no_grad():
                                # 지터 폭을 여러 개로 잰다. δ 가 문서 간 자연 변이보다
                                # 작으면 「반응 없음」이 당연한 결과가 되어 해석이 불가능하다.
                                # 실측 자연 변이(0~1000 좌표계) 표준편차 평균 66.8.
                                for d in a.jitter:
                                    p_j = float(sc(ov={"bbox": A.jitter_boxes(
                                        inputs["bbox"], gold_tk, d, gen)}))
                                    rec["faith"][f"layout_jitter@{d}"] = p_full - p_j
                                other = [t for t in range(len(spans))
                                         if spans[t] and t not in gold_tk][: len(gold_tk)]
                                p_swap = float(sc(ov={"bbox": A.swap_boxes(
                                    inputs["bbox"], gold_tk, other)})) if other else None
                                p_occ = None
                                if "image" in inputs:
                                    bn = [e.boxes[i] for i in sorted(gold_widx)
                                          if i < len(e.boxes)]
                                    p_occ = float(sc(ov={"image": A.occlude_image(
                                        inputs["image"], bn)}))
                            rec["faith"]["layout_swap"] = (
                                p_full - p_swap if p_swap is not None else None)
                            rec["faith"]["image_occlusion"] = (
                                p_full - p_occ if p_occ is not None else None)

                        # --- NAOPC — 위에서 (문서, 필드)당 한 번 구한 한계값을 쓴다 ---
                        # 논문 정의대로 전체 순서를 훑는 AOPC를 따로 계산한다.
                        # 상한·하한이 그 정의 위에서 나오므로 k 격자 값과 섞으면 안 된다.
                        if lo is not None:
                            # 보호 집합은 순서에서 뺀다. 상한·하한도 같은 후보 집합
                            # 위에서 구했으므로 자가 일치한다.
                            order = topk_excluding(units, scores, len(units), gold_widx)
                            if a.naopc_max_steps:
                                order = order[: a.naopc_max_steps]
                            a_full = A.aopc_full(eval_masked, p_full, order)
                            rec["faith"]["aopc_full"] = a_full
                            rec["faith"]["aopc_lower"] = lo
                            rec["faith"]["aopc_upper"] = up
                            rec["faith"]["naopc"] = naopc(a_full, lo, up)

                        emit(rec)

        if (n_doc + 1) % 10 == 0:
            print(f"  {n_doc+1}/{len(ex)} 문서 완료", flush=True)

    f_out.close()
    if f_phi is not None:
        f_phi.close()
    print(f"\n완료 → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
