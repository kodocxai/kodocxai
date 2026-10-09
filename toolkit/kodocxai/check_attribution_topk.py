"""귀속 기법의 최상위 단위가 어디를 가리키는가 (원고 4.2절 근거).

표 4에서 어떤 기법의 hit@1 이 0.000 으로 나오면 두 가지를 갈라야 한다.
  A: 최상위가 특수토큰·시각토큰 등 텍스트가 아닌 곳이다 → 구현·집계 문제
  B: 텍스트를 짚기는 하는데 근거와 무관한 곳이다        → 실제 결과

기법별로 상위 5개 단위의 **종류**와 **어절 번호**를 찍어 준다. 최상위가 늘 같은 어절이면
위치 편향을 의심할 것.

    python check_attribution_topk.py [--model runs\seed1\last] [--n 6]
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kodx_data import build_examples, load_processor          # noqa: E402
from kodx_units import subword_char_spans, aggregate, units_to_word_idx  # noqa: E402
import kodx_attrib as A                                       # noqa: E402

PILOT = Path(r"D:\AIHub데이터\_pilot")
EXP = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(EXP / "runs/seed1/last"))
    ap.add_argument("--images", default=str(EXP / "data/pilot_images224"))
    ap.add_argument("--n", type=int, default=6)
    a = ap.parse_args()

    from transformers import AutoModelForTokenClassification

    proc = load_processor(a.model, apply_ocr=False)
    tok = proc.tokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = AutoModelForTokenClassification.from_pretrained(
        a.model, output_attentions=True).to(dev).eval()
    special = set(tok.all_special_ids)

    ex = [e for e in build_examples(PILOT / "docs.jsonl", Path(a.images),
                                    annotations=PILOT / "annotations_park.jsonl")
          if e.gold_spans][: a.n]

    kinds: Counter = Counter()
    top_word: Counter = Counter()
    hit = miss = 0

    for e in ex:
        enc = proc(Image.open(e.image_path).convert("RGB"), e.words, boxes=e.boxes,
                   truncation=True, padding="max_length", max_length=512,
                   return_offsets_mapping=True, return_tensors="pt")
        offsets = enc.pop("offset_mapping")
        spans = subword_char_spans(tok, e.words, {"word_ids": enc.word_ids(),
                                                  "offset_mapping": offsets[0]})
        inputs = {k: v.to(dev) for k, v in enc.items()}
        field, gold = next(iter(e.gold_spans.items()))
        gw = set(gold["word_idx"])
        tgt = [t for t, s in enumerate(spans) if s and s[0] in gw]
        if not tgt:
            continue
        with torch.no_grad():
            att = m(**inputs, output_attentions=True).attentions
        phi = A.attention_rollout(att, tgt[0]).detach().float().cpu().tolist()

        ids = inputs["input_ids"][0].tolist()
        order = sorted(range(len(phi)), key=lambda i: -phi[i])[:5]
        print(f"\n=== {e.doc_id} · 필드 {field} · 정답 어절 {sorted(gw)[:6]}")
        for r, i in enumerate(order):
            kind = ("시각토큰" if i >= len(spans)
                    else "특수토큰" if ids[i] in special
                    else "패딩" if spans[i] is None
                    else f"어절{spans[i][0]}")
            txt = tok.convert_ids_to_tokens([ids[i]])[0] if i < len(ids) else "-"
            print(f"   {r+1}위 idx={i:3d} {kind:9s} {txt!r:14s} phi={phi[i]:.5f}")
            if r == 0:
                kinds["어절" if kind.startswith("어절") else kind] += 1
                if kind.startswith("어절"):
                    top_word[int(kind[2:])] += 1

        sc, un = aggregate(phi, spans, e.words, "eojeol")
        t1 = units_to_word_idx(un, sc, 1)
        print(f"   어절 집계 top-1 {t1} · 정답 안에 있는가: {bool(set(t1) & gw)}")
        hit += bool(set(t1) & gw)
        miss += not bool(set(t1) & gw)

    print("\n=== 요약 ===")
    print(f"최상위 단위의 종류: {dict(kinds)}")
    print(f"최상위 어절 번호  : {dict(top_word.most_common(5))}")
    print(f"top-1 이 정답 안  : {hit} / {hit + miss}")
    if top_word and top_word.most_common(1)[0][1] == hit + miss:
        w = top_word.most_common(1)[0][0]
        print(f"\n⚠ 모든 문서에서 최상위가 **어절 {w}** 이다 — 위치 편향이다.")
        print("  구현 오류가 아니라 기법의 성질일 수 있다. attention rollout 은 잔차 연결")
        print("  때문에 앞쪽 토큰에 질량이 쏠리는 것으로 알려져 있다. 원고 4.2절 참조.")


if __name__ == "__main__":
    main()
