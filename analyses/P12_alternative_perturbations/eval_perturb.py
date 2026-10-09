# -*- coding: utf-8 -*-
"""P12 대체 교란 전략 비교 (MDPI R1-3·R5-2, 10-02 발주).

현행 충실도는 [MASK] 치환(분포내 유지) 하나로 측정했다. 리뷰 지적: 교란 전략에
결론이 의존하는지 확인하라. 두 변형으로 어절 수준 COMP/SUFF@k·AOPC 를 재산출한다.

전략 (원고 인용 명칭)
  delete  : 토큰 삭제 - 선택 어절의 토큰을 시퀀스에서 제거(레이아웃 상자 동반 제거),
            남은 토큰을 앞으로 당기고 뒤를 PAD. 특수 토큰(CLS/SEP)은 항상 보존.
            타깃 토큰 위치는 삭제 후 좌표로 재매핑. 미보호에서 타깃이 전부 삭제되면
            p_perturbed = 0 으로 정의(예측 대상 소멸 = 최대 낙폭).
  randsub : 어휘 내 무작위 토큰 치환 - 선택 어절의 토큰을 특수 토큰이 아닌 무작위
            어휘 토큰으로 치환(위치·레이아웃 불변). RNG 는 시드 고정.

각 전략 × 보호 유무(v2 보호 / v1 미보호) × 기법 3종(정본 phi 재사용) × 시드 3.
집계 수준 = 어절·전체. p_full·예측 라벨·순위 경로는 v2 와 동일.

사용
    python eval_perturb.py --model ..\runs\seed1\last --phi ..\results\phi_seed1.jsonl \
        --out P12_seed1.jsonl --seed 1
산출물은 이 폴더에만. frozen 자산·정본 무접촉.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

EXP = Path(r"D:\AIHub데이터\_exp")
sys.path.insert(0, str(EXP))

from kodx_data import build_examples, load_processor                 # noqa: E402
from kodx_units import Tagger, aggregate, subword_char_spans         # noqa: E402
from kodx_metrics import aopc, comp_suff                             # noqa: E402
import kodx_attrib as A                                              # noqa: E402
from kodx_eval import token_idx_for_words, topk_excluding            # noqa: E402

KS = [1, 3, 5, 10]
METHODS = ["ig", "rollout", "random"]
STRATS = ["delete", "randsub"]
PROTS = ["protected", "unprotected"]


def delete_tokens(inputs, drop: set, pad_id: int):
    """drop 위치 토큰 제거 후 좌측 압축 + PAD. (새 inputs, 구→신 위치 매핑) 반환."""
    ids = inputs["input_ids"][0]
    T = ids.size(0)
    keep = [t for t in range(T) if t not in drop]
    mapping = {old: new for new, old in enumerate(keep)}
    new = {}
    ni = torch.full_like(ids, pad_id)
    ni[: len(keep)] = ids[keep]
    new["input_ids"] = ni.unsqueeze(0)
    bb = inputs["bbox"][0]
    nb = torch.zeros_like(bb)
    nb[: len(keep)] = bb[keep]
    new["bbox"] = nb.unsqueeze(0)
    am = torch.zeros_like(inputs["attention_mask"][0])
    am[: len(keep)] = 1
    new["attention_mask"] = am.unsqueeze(0)
    for k, v in inputs.items():
        if k not in new:
            if v.dim() >= 2 and v.size(1) == T:          # 토큰 축 텐서
                nv = v[0].clone()
                nv[: len(keep)] = v[0][keep]
                nv[len(keep):] = 0
                new[k] = nv.unsqueeze(0)
            else:
                new[k] = v                               # image 등
    return new, mapping


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--phi", required=True)
    ap.add_argument("--docs", default=str(Path(r"D:\AIHub데이터\_pilot\docs.jsonl")))
    ap.add_argument("--images", default=str(EXP / "data" / "pilot_images224"))
    ap.add_argument("--annotations",
                    default=str(Path(r"D:\AIHub데이터\_pilot\annotations_park.jsonl")))
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    # 표본 축소(10-02): 전 표본 300×3시드는 마감(10/6) 내 불가 판정 → 고정 표본
    # 100문서(기계산 34 + RNG 20261002 표집 66, 시드 공통). P7 원안(60~100건)과 정합.
    ap.add_argument("--doc-list", default=None,
                    help="평가할 doc_id 목록 JSON({'docs': [...]}). 지정 시 그 문서만")
    a = ap.parse_args()

    from transformers import AutoModelForTokenClassification
    from PIL import Image

    phi_map = {}
    for line in Path(a.phi).open(encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            phi_map[(r["doc_id"], r["field"], r["method"])] = r["phi"]

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = load_processor(a.model, apply_ocr=False)
    tok = proc.tokenizer
    pad_id = tok.pad_token_id
    vocab = tok.vocab_size
    special_ids = set(tok.all_special_ids)
    model = AutoModelForTokenClassification.from_pretrained(a.model).to(dev).eval()
    tagger = Tagger()
    gen = torch.Generator(device="cpu").manual_seed(a.seed + 20261002)

    ex = build_examples(Path(a.docs), Path(a.images), Path(a.annotations))
    if a.doc_list:
        allow = set(json.loads(Path(a.doc_list).read_text(encoding="utf-8"))["docs"])
        ex = [e for e in ex if e.doc_id in allow]
    if a.limit:
        ex = ex[: a.limit]
    print(f"평가 문서 {len(ex)}건 / 장치 {dev}")

    out = Path(a.out)
    done = set()
    if out.exists():
        for line in out.open(encoding="utf-8"):
            try:
                r = json.loads(line)
                done.add((r["doc_id"], r["field"], r["method"], r["strategy"], r["protection"]))
            except ValueError:
                continue
        print(f"기존 조합 {len(done)}개 건너뜀")
    f_out = out.open("a", encoding="utf-8")

    def emit(rec):
        f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f_out.flush(); os.fsync(f_out.fileno())

    def rand_ids(n):
        ids = torch.randint(0, vocab, (n,), generator=gen).tolist()
        return [i if i not in special_ids else (i + 7) % vocab for i in ids]

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

        def score_at(inp, tpos, labels):
            if not tpos:
                return 0.0
            with torch.no_grad():
                return float(A.field_score(model(**inp).logits, tpos, labels))

        for field, gold in e.gold_spans.items():
            gold_widx = set(gold["word_idx"])
            if not gold_widx:
                continue
            tgt_tokens = token_idx_for_words(spans, gold_widx)
            if not tgt_tokens:
                continue
            label_ids = [int(pred_ids[t]) for t in tgt_tokens]
            p_full = score_at(inputs, tgt_tokens, label_ids)
            special_tok = [t for t, s in enumerate(spans) if s is None]

            def perturb_score(sel_words, strat, keep_mode, protect_gold):
                """sel_words 를 교란(COMP) 또는 그 밖을 교란(SUFF)한 점수."""
                sel_tk = set(token_idx_for_words(spans, set(sel_words)))
                if keep_mode:                     # SUFF: 나머지를 교란
                    keep = sel_tk | set(special_tok)
                    if protect_gold:
                        keep |= set(tgt_tokens)
                    target = [t for t, s in enumerate(spans)
                              if s is not None and t not in keep]
                else:                             # COMP: 선택을 교란
                    target = sorted(sel_tk)
                if strat == "randsub":
                    ids = inputs["input_ids"].clone()
                    for t, rid in zip(target, rand_ids(len(target))):
                        ids[0, t] = rid
                    return score_at({**inputs, "input_ids": ids},
                                    tgt_tokens, label_ids)
                # delete
                new, mapping = delete_tokens(inputs, set(target), pad_id)
                tpos, labels = [], []
                for t, l in zip(tgt_tokens, label_ids):
                    if t in mapping:
                        tpos.append(mapping[t]); labels.append(l)
                return score_at(new, tpos, labels)

            for method in METHODS:
                phi = phi_map.get((e.doc_id, field, method))
                if phi is None:
                    continue
                scores, units = aggregate(phi, spans, e.words, "eojeol",
                                          tagger, "all")
                if not scores:
                    continue
                for strat in STRATS:
                    for prot in PROTS:
                        key = (e.doc_id, field, method, strat, prot)
                        if key in done:
                            continue
                        protect = (prot == "protected")
                        comps, suffs = [], []
                        for k in KS:
                            if protect:
                                widx = topk_excluding(units, scores, k, gold_widx)
                            else:
                                widx = topk_excluding(units, scores, k, set())
                            if not widx:
                                comps.append(None); suffs.append(None); continue
                            p_wo = perturb_score(widx, strat, False, protect)
                            p_only = perturb_score(widx, strat, True, protect)
                            cs = comp_suff(p_full, p_wo, p_only)
                            comps.append(cs["comp"]); suffs.append(cs["suff"])
                        emit({"doc_id": e.doc_id, "field": field, "method": method,
                              "strategy": strat, "protection": prot,
                              "level": "eojeol", "mask": "all", "p_full": p_full,
                              "faith": {"comp@k": comps, "suff@k": suffs,
                                        "aopc_comp": aopc(comps)}})

        if (n_doc + 1) % 25 == 0:
            print(f"  {n_doc+1}/{len(ex)} 문서 완료", flush=True)

    f_out.close()
    print(f"완료 → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
