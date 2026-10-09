# -*- coding: utf-8 -*-
"""P16 phi 결손 보충 (2026-10-07).

원인: kodx_eval 의 phi 기록이 무flush라 중단 시 버퍼 꼬리 유실(평가 레코드는
flush+fsync라 무결). 레코드에는 있는데 phi 에 없는 (doc, field, method)만
재계산해 추가한다. ig·rollout 은 결정적 재계산(값 동일), random 은 재시드
재추출(실행노트의 RNG 재시드 항목과 동일 성격 - 기준선 유효성 무영향).

사용: python backfill_phi.py --seed 4   (그리고 5)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

EXP = Path(r"D:\AIHub데이터\_exp")
P16 = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))

from kodx_data import build_examples, load_processor                 # noqa: E402
from kodx_units import subword_char_spans                            # noqa: E402
import kodx_attrib as A                                              # noqa: E402
from kodx_eval import token_idx_for_words                            # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    a = ap.parse_args()
    s = a.seed

    have = set()
    pf = P16 / f"phi_seed{s}.jsonl"
    for line in pf.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            have.add((r["doc_id"], r["field"], r["method"]))
    need = set()
    for line in (P16 / f"seed{s}_v2.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            k = (r["doc_id"], r["field"], r["method"])
            if k not in have:
                need.add(k)
    print(f"seed{s}: 결손 {len(need)}건")
    if not need:
        return 0

    from transformers import AutoModelForTokenClassification
    from PIL import Image

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = str(EXP / "runs" / f"seed{s}" / "last")
    proc = load_processor(model_path, apply_ocr=False)
    tok = proc.tokenizer
    mask_id = tok.mask_token_id
    model = AutoModelForTokenClassification.from_pretrained(model_path).to(dev).eval()
    embed_fn = model.get_input_embeddings()
    gen = torch.Generator(device="cpu").manual_seed(s + 20261007)

    need_docs = {d for d, _, _ in need}
    ex = [e for e in build_examples(
        Path(r"D:\AIHub데이터\_pilot\docs.jsonl"), EXP / "data" / "pilot_images224",
        Path(r"D:\AIHub데이터\_pilot\annotations_park.jsonl")) if e.doc_id in need_docs]

    out = pf.open("a", encoding="utf-8")
    n_done = 0
    for e in ex:
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

        def score_fn(m, inp, inputs_embeds=None, token_idx=None, label_ids=None):
            kw = dict(inp)
            if inputs_embeds is not None:
                kw.pop("input_ids", None)
                kw["inputs_embeds"] = inputs_embeds
            return A.field_score(m(**kw).logits, token_idx, label_ids)

        for field, gold in e.gold_spans.items():
            gw = set(gold["word_idx"])
            if not gw:
                continue
            tt = token_idx_for_words(spans, gw)
            if not tt:
                continue
            ll = [int(pred_ids[t]) for t in tt]
            for method in ("ig", "rollout", "random"):
                if (e.doc_id, field, method) not in need:
                    continue
                if method == "ig":
                    phi = A.integrated_gradients(
                        model, inputs, lambda ids: embed_fn(ids),
                        lambda m, inp, inputs_embeds: score_fn(
                            m, inp, inputs_embeds, tt, ll),
                        steps=32, baseline_token_id=mask_id).abs()
                elif method == "rollout":
                    with torch.no_grad():
                        att = model(**inputs, output_attentions=True).attentions
                    phi = A.attention_rollout(att, tt[0])
                else:
                    phi = A.random_attribution(base_logits.shape[1], gen)
                out.write(json.dumps(
                    {"doc_id": e.doc_id, "field": field, "method": method,
                     "phi": [round(v, 6) for v in phi.detach().float().cpu().tolist()]},
                    ensure_ascii=False) + "\n")
                out.flush()
                n_done += 1
    out.close()
    print(f"보충 {n_done}건 완료")
    return 0


if __name__ == "__main__":
    sys.exit(main())
