# -*- coding: utf-8 -*-
"""P5 덤프 - 예측 스팬(absent 필드 포함) + 귀속 top-10 어절 (검토서 §2.7).

kie_seed{N}.json 은 인스턴스별 카운트만 남겨 ①absent 필드에 대한 예측(spurious)
②예측 스팬의 위치(error-grounding에 필요)가 없다. 여기서 시드당 300 순전파로 둘을
덤프한다. 디코딩은 eval_kie.decode_spans 를 그대로 임포트(정의 동일 보장).

레코드 2종 (JSONL)
  {"type":"pred", doc_id, field, gold:[어절], pred:[어절], in_window:bool}
      - gold 빈 리스트 = absent 필드(spurious 판정용)
  {"type":"topk", doc_id, field, method, top10:[어절 순서대로]}
      - 정본 phi 를 어절 수준(전체)으로 재집계한 무보호 상위 10 (타당성 경로와 동일,
        units_to_word_idx). error-grounding 에서 앞 1·5개를 잘라 쓴다.

사용
    python dump_pred.py --model ..\runs\seed1\last --phi ..\results\phi_seed1.jsonl \
        --out P5_pred_seed1.jsonl
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

from kodx_data import FIELDS, build_examples, load_processor         # noqa: E402
from kodx_units import Tagger, aggregate, subword_char_spans, units_to_word_idx  # noqa: E402
from eval_kie import decode_spans                                    # noqa: E402

METHODS = ["ig", "rollout", "random"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--phi", required=True)
    ap.add_argument("--docs", default=str(Path(r"D:\AIHub데이터\_pilot\docs.jsonl")))
    ap.add_argument("--images", default=str(EXP / "data" / "pilot_images224"))
    ap.add_argument("--annotations",
                    default=str(Path(r"D:\AIHub데이터\_pilot\annotations_park.jsonl")))
    ap.add_argument("--out", required=True)
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
    model = AutoModelForTokenClassification.from_pretrained(a.model).to(dev).eval()
    tagger = Tagger()

    ex = build_examples(Path(a.docs), Path(a.images), Path(a.annotations))
    print(f"평가 문서 {len(ex)}건 / 장치 {dev}")

    f_out = Path(a.out).open("w", encoding="utf-8")

    def emit(rec):
        f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    for n, e in enumerate(ex):
        # ⚠ 여기서는 gold_spans 유무와 무관하게 전 문서를 돈다 - absent 전용 문서
        # (5필드 모두 absent 1건)에서도 spurious 예측을 세야 한다.
        img = Image.open(e.image_path).convert("RGB")
        enc = proc(img, e.words, boxes=e.boxes, truncation=True,
                   padding="max_length", max_length=512,
                   return_offsets_mapping=True, return_tensors="pt")
        offs = enc.pop("offset_mapping")
        spans = subword_char_spans(tok, e.words, {"word_ids": enc.word_ids(),
                                                  "offset_mapping": offs[0]})
        inputs = {k: v.to(dev) for k, v in enc.items()}
        with torch.no_grad():
            pred_ids = model(**inputs).logits.argmax(-1)[0].cpu()
        pred = decode_spans(pred_ids, spans, len(e.words))
        covered = {s[0] for s in spans if s is not None}

        for f in FIELDS:
            g = e.gold_spans.get(f)
            gw = sorted(set(g["word_idx"])) if g and g.get("word_idx") else []
            in_win = bool(set(gw) & covered) if gw else True
            emit({"type": "pred", "doc_id": e.doc_id, "field": f,
                  "gold": gw, "pred": sorted(pred.get(f, set())),
                  "in_window": in_win})

        for f, g in e.gold_spans.items():
            if not g.get("word_idx"):
                continue
            for m in METHODS:
                phi = phi_map.get((e.doc_id, f, m))
                if phi is None:
                    continue
                scores, units = aggregate(phi, spans, e.words, "eojeol",
                                          tagger, "all")
                if not scores:
                    continue
                emit({"type": "topk", "doc_id": e.doc_id, "field": f,
                      "method": m, "top10": units_to_word_idx(units, scores, 10)})

        if (n + 1) % 50 == 0:
            print(f"  {n+1}/{len(ex)}", flush=True)

    f_out.close()
    print(f"완료 → {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
