# -*- coding: utf-8 -*-
"""P11 rollout span 수준 일관 타깃 재평가 (MDPI R1 에디터 포인트 1, 10-02 발주).

현행 rollout 타깃 = 정답 스팬 「첫 토큰」(단순화, 원고에 한계로 명시). 에디터 요구
"evaluate both methods against a consistent span-level target"에 따라 타깃을 스팬
수준으로 확장한 2변형을 재계산한다. IG 는 재계산하지 않는다(이미 스팬 수준 - field_score
기하평균에 대한 기울기라 대조 유지).

변형 (rollout 행렬 R 은 문서당 1회 계산 - 타깃과 무관한 층별 어텐션 누적)
  rollout_mean : phi = mean_{t in span} R[t]   (스팬 토큰 행 평균)
  rollout_sum  : phi = sum_{t in span} R[t]    (스팬 토큰 행 합)
행렬 수식은 kodx_attrib.attention_rollout 와 동일(헤드 평균 → +I → 행 정규화 → 층 곱).

측정은 v2 프로토콜 그대로: 어절 수준·전체 마스킹, 정답 스팬 보호, [MASK] 치환,
p_full = 예측 라벨 기하평균. 산출 = 표 5 지표(IoU@1/5·F1@5·Hit@1) + COMP/SUFF@k·AOPC.

사용
    python eval_rollout_span.py --model ..\runs\seed1\last --out P11_seed1.jsonl
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
from kodx_units import Tagger, aggregate, subword_char_spans, units_to_word_idx  # noqa: E402
from kodx_metrics import aopc, comp_suff, plausibility               # noqa: E402
import kodx_attrib as A                                              # noqa: E402
from kodx_eval import token_idx_for_words, topk_excluding            # noqa: E402

KS = [1, 3, 5, 10]
VARIANTS = ["rollout_mean", "rollout_sum"]


def rollout_matrix(attentions) -> torch.Tensor:
    """kodx_attrib.attention_rollout 와 동일 수식, 전체 행렬 반환."""
    result = None
    for Att in attentions:
        a = Att[0].mean(0)
        a = a + torch.eye(a.size(0), device=a.device)
        a = a / a.sum(-1, keepdim=True)
        result = a if result is None else a @ result
    return result                                           # (T, T)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--docs", default=str(Path(r"D:\AIHub데이터\_pilot\docs.jsonl")))
    ap.add_argument("--images", default=str(EXP / "data" / "pilot_images224"))
    ap.add_argument("--annotations",
                    default=str(Path(r"D:\AIHub데이터\_pilot\annotations_park.jsonl")))
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    from transformers import AutoModelForTokenClassification
    from PIL import Image

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = load_processor(a.model, apply_ocr=False)
    tok = proc.tokenizer
    mask_id = tok.mask_token_id
    model = AutoModelForTokenClassification.from_pretrained(a.model).to(dev).eval()
    tagger = Tagger()

    ex = build_examples(Path(a.docs), Path(a.images), Path(a.annotations))
    if a.limit:
        ex = ex[: a.limit]
    print(f"평가 문서 {len(ex)}건 / 장치 {dev}")

    out = Path(a.out)
    done = set()
    if out.exists():
        for line in out.open(encoding="utf-8"):
            try:
                r = json.loads(line)
                done.add((r["doc_id"], r["field"], r["method"]))
            except ValueError:
                continue
        print(f"기존 조합 {len(done)}개 건너뜀")
    f_out = out.open("a", encoding="utf-8")

    def emit(rec):
        f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f_out.flush(); os.fsync(f_out.fileno())

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
            o = model(**inputs, output_attentions=True)
        pred_ids = o.logits.argmax(-1)[0]
        R = rollout_matrix(o.attentions)                     # 문서당 1회
        del o

        def sc(ov=None):
            inp = dict(inputs) if ov is None else {**inputs, **ov}
            return A.field_score(model(**inp).logits, tgt_tokens, label_ids)

        for field, gold in e.gold_spans.items():
            gold_widx = set(gold["word_idx"])
            if not gold_widx:
                continue
            tgt_tokens = token_idx_for_words(spans, gold_widx)
            if not tgt_tokens:
                continue
            label_ids = [int(pred_ids[t]) for t in tgt_tokens]
            with torch.no_grad():
                p_full = float(sc())
            gold_tk = token_idx_for_words(spans, gold_widx)
            special = [t for t, s in enumerate(spans) if s is None]

            rows = R[tgt_tokens]                             # (|span|, T)
            for method in VARIANTS:
                if (e.doc_id, field, method) in done:
                    continue
                phi = (rows.mean(0) if method == "rollout_mean"
                       else rows.sum(0)).detach().float().cpu().tolist()
                scores, units = aggregate(phi, spans, e.words, "eojeol",
                                          tagger, "all")
                if not scores:
                    continue
                rec = {"doc_id": e.doc_id, "field": field, "method": method,
                       "level": "eojeol", "mask": "all", "p_full": p_full,
                       "plaus": {}, "faith": {}}
                for k in KS:
                    widx = units_to_word_idx(units, scores, k)
                    rec["plaus"][str(k)] = plausibility(
                        widx, sorted(gold_widx), e.raw_boxes,
                        ignore_word_idx=e.ignore_word_idx)
                comps, suffs = [], []
                for k in KS:
                    widx = set(topk_excluding(units, scores, k, gold_widx))
                    tk = token_idx_for_words(spans, widx)
                    if not tk:
                        comps.append(None); suffs.append(None); continue
                    with torch.no_grad():
                        p_wo = float(sc(ov={"input_ids": A.mask_tokens(
                            inputs["input_ids"], tk, mask_id)}))
                        p_only = float(sc(ov={"input_ids": A.keep_only(
                            inputs["input_ids"], sorted(set(tk) | set(gold_tk)),
                            mask_id, special)}))
                    cs = comp_suff(p_full, p_wo, p_only)
                    comps.append(cs["comp"]); suffs.append(cs["suff"])
                rec["faith"]["comp@k"] = comps
                rec["faith"]["suff@k"] = suffs
                rec["faith"]["aopc_comp"] = aopc(comps)
                emit(rec)

        del R
        if (n_doc + 1) % 25 == 0:
            print(f"  {n_doc+1}/{len(ex)} 문서 완료", flush=True)

    f_out.close()
    print(f"완료 → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
