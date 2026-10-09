# -*- coding: utf-8 -*-
"""값 스팬 포함 텍스트 마스킹 민감도 (2차 검수 N2 → E16, 저자 채택 2026-08-17).

표 6 의 Text 열은 정답 보호 때문에 **문맥만** 마스킹한다(COMP@10). 이 스크립트는
보호를 걷고 **값 스팬 자체**를 마스킹했을 때의 확률 낙폭을 필드별 정식 지표로 낸다.

  d_value     = p_full - p(값 스팬 [MASK])           : 값만
  d_value_ctx = p_full - p(값 스팬 ∪ IG top-10 [MASK]): 값 + 문맥 (표 6 Text 열과 같은
                문맥 집합 - IG 어절 상위 10, 정답 제외 후 충원)

측정 관례는 kodx_eval.py v2 와 동일하다 - 타깃은 정답 스팬 위치의 **교란 전 예측
라벨**(무예측이면 O), 점수는 기하평균(field_score), 마스킹은 [MASK] 치환(분포내).
표본도 v2 와 같은 1,166 인스턴스다(φ 저장분과 교집합).

순환 문제(v1의 교훈)와 무엇이 다른가: 이 지표는 충실도 순위 비교용이 아니라
**모달 민감도**(값 텍스트를 잃으면 예측이 얼마나 무너지는가)다. 「정의상 당연」한
낙폭 그 자체가 측정 대상이므로 보호가 필요 없다 - 표 6 의 지터·occlusion 과 같은
지위다. 귀속 기법 간 비교에는 쓰면 안 된다.

사용
    python eval_textmask.py [--seeds 1 2 3]      # jsonl 3개 + 요약 md
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kodx_data import build_examples, load_processor                 # noqa: E402
from kodx_units import aggregate, subword_char_spans                 # noqa: E402
import kodx_attrib as A                                              # noqa: E402

PILOT = Path(r"D:\AIHub데이터\_pilot")
RES = Path(r"D:\AIHub데이터\_exp\results")
RUNS = Path(r"D:\AIHub데이터\_exp\runs")
OUT_MD = RES / "값마스킹_민감도.md"
FIELDS = ["제목", "수신", "문서번호", "시행일자", "발신명의"]
CTX_K = 10                     # 표 6 Text(context) 열과 같은 k


def token_idx_for_words(spans, word_idx: set) -> list[int]:
    return [t for t, s in enumerate(spans) if s and s[0] in word_idx]


def topk_excluding(units, scores, k, protect):
    """kodx_eval.py 와 동일 - 보호 집합은 순위에서 빼고 다음 순위로 채운다."""
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


def load_phi_ig(seed: int) -> dict:
    got = {}
    for line in (RES / f"phi_seed{seed}.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        if r["method"] == "ig":
            got[(r["doc_id"], r["field"])] = r["phi"]
    return got


def run_seed(seed: int, ex, dev: str) -> Path:
    out = RES / f"textmask_seed{seed}.jsonl"
    done = set()
    if out.exists():
        for line in out.open(encoding="utf-8"):
            try:
                r = json.loads(line)
                done.add((r["doc_id"], r["field"]))
            except ValueError:
                continue
        if done:
            print(f"  seed{seed}: 기왕 계산 {len(done)}건 건너뜀")

    from transformers import AutoModelForTokenClassification
    model_dir = RUNS / f"seed{seed}" / "last"
    proc = load_processor(str(model_dir), apply_ocr=False)
    tok = proc.tokenizer
    mask_id = tok.mask_token_id
    model = AutoModelForTokenClassification.from_pretrained(str(model_dir)).to(dev).eval()
    phi = load_phi_ig(seed)

    from PIL import Image
    f_out = out.open("a", encoding="utf-8")
    n_done = 0
    for e in ex:
        if not e.gold_spans:
            continue
        keys = [(f, g) for f, g in e.gold_spans.items()
                if (e.doc_id, f) in phi and (e.doc_id, f) not in done]
        if not keys:
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

        for field, gold in keys:
            gold_widx = set(gold["word_idx"])
            if not gold_widx:
                continue
            tgt_tokens = token_idx_for_words(spans, gold_widx)
            if not tgt_tokens:
                continue
            label_ids = [int(pred_ids[t]) for t in tgt_tokens]

            def sc(ids):
                with torch.no_grad():
                    lg = model(**{**inputs, "input_ids": ids}).logits
                return float(A.field_score(lg, tgt_tokens, label_ids))

            p_full = sc(inputs["input_ids"])
            p_val = sc(A.mask_tokens(inputs["input_ids"], tgt_tokens, mask_id))

            scores, units = aggregate(phi[(e.doc_id, field)], spans, e.words, "eojeol")
            ctx_w = topk_excluding(units, scores, CTX_K, gold_widx)
            ctx_tk = token_idx_for_words(spans, set(ctx_w))
            p_both = sc(A.mask_tokens(inputs["input_ids"],
                                      sorted(set(tgt_tokens) | set(ctx_tk)), mask_id))

            f_out.write(json.dumps({
                "doc_id": e.doc_id, "field": field, "p_full": p_full,
                "d_value": p_full - p_val,
                "d_value_ctx": p_full - p_both,
            }, ensure_ascii=False) + "\n")
            f_out.flush()
            n_done += 1
        if n_done and n_done % 200 == 0:
            print(f"  seed{seed}: {n_done}건", flush=True)
    f_out.close()
    print(f"  seed{seed}: 신규 {n_done}건 → {out.name}")
    return out


def ctx_only_from_v2(seed: int) -> dict:
    """표 6 Text 열 재현 - v2 기록의 IG·어절·COMP@10 (문맥만)."""
    got = {}
    for line in (RES / f"seed{seed}_v2.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        if r["method"] == "ig" and r["level"] == "eojeol" and r["mask"] == "all":
            c = (r["faith"].get("comp@k") or [None] * 4)[-1]
            if c is not None:
                got[(r["doc_id"], r["field"])] = c
    return got


def summarize(seeds) -> None:
    per = {}                                   # seed → field → {지표: [값]}
    for s in seeds:
        acc = defaultdict(lambda: defaultdict(list))
        for line in (RES / f"textmask_seed{s}.jsonl").open(encoding="utf-8"):
            r = json.loads(line)
            acc[r["field"]]["d_value"].append(r["d_value"])
            acc[r["field"]]["d_value_ctx"].append(r["d_value_ctx"])
            acc["(전체)"]["d_value"].append(r["d_value"])
            acc["(전체)"]["d_value_ctx"].append(r["d_value_ctx"])
        for (d, f), c in ctx_only_from_v2(s).items():
            acc[f]["ctx_only"].append(c)
            acc["(전체)"]["ctx_only"].append(c)
        per[s] = acc

    def cell(field, key, digits=4):
        means = [st.mean(per[s][field][key]) for s in seeds if per[s][field][key]]
        if not means:
            return "-"
        sd = st.pstdev(means) if len(means) > 1 else 0.0
        return f"{st.mean(means):.{digits}f} ± {sd:.{digits}f}"

    n = round(st.mean(sum(len(v["d_value"]) for f, v in per[s].items() if f != "(전체)")
                      for s in seeds))
    L = []
    add = L.append
    add("# 값 스팬 포함 텍스트 마스킹 민감도 (E16)")
    add("")
    add("> `_exp\\eval_textmask.py` 가 만든다. **손으로 고치지 말 것.**")
    add("> 2차 검수 N2 대응(저자 채택 08-17). 시드 내 인스턴스 평균 후 시드 간 평균±표준편차.")
    add(f"> 표본 {n}건/시드 (v2 와 동일 집합) · 문맥 = IG 어절 상위 {CTX_K}(정답 제외 후 충원).")
    add("")
    add("측정: 교란 전 예측 라벨을 고정한 필드 확률(기하평균)의 낙폭. [MASK] 치환(분포내).")
    add("**귀속 기법 간 비교용이 아니다** - 값 텍스트 모달의 의존도를 재는 표 6 계열 지표다.")
    add("")
    add("| 필드 | 문맥만 (기존 표 6 Text) | **값만 (신규)** | **값+문맥 (신규)** |")
    add("|---|---:|---:|---:|")
    for f in FIELDS + ["(전체)"]:
        add(f"| {f} | {cell(f, 'ctx_only')} | **{cell(f, 'd_value')}** | "
            f"**{cell(f, 'd_value_ctx')}** |")
    add("")
    add("## 읽는 법")
    add("")
    add("- **값만**이 크면 예측이 값 텍스트 자체에 의존한다는 뜻이다. 문맥만(보호 하)과의")
    add("  대비가 「보호 프로토콜이 무엇을 측정에서 분리했는가」를 정량화한다.")
    add("- **값+문맥**은 텍스트 모달 전체를 걷어낸 상한 - 지터·occlusion(표 6)과 나란히")
    add("  읽으면 모달 의존의 전모가 된다.")
    add("- 값만 낙폭이 1 에 가까우면 BIO 라벨링의 구조상 당연한 결과가 포함된다(3.3.2 의")
    add("  순환 논지). 그래서 이 지표는 충실도 순위가 아니라 **모달 민감도**로만 쓴다.")
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"생성: {OUT_MD}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ex = build_examples(PILOT / "docs.jsonl", PILOT / "images",
                        annotations=PILOT / "annotations_park.jsonl")
    print(f"평가 문서 {len(ex)}건 / 장치 {dev}")
    for s in a.seeds:
        run_seed(s, ex, dev)
    summarize(a.seeds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
