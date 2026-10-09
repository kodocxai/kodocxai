# -*- coding: utf-8 -*-
"""P15 귀속 기법 2종 추가 - Gradient×Input · 가림(leave-one-out) (R5-1, 10-05 발주).

기법 정의
  gxi  : 정답 스팬 확률(field_score, IG와 동일 타깃)의 임베딩 기울기 × 임베딩,
         차원 합 후 절댓값(IG의 abs 관례와 통일). 1역전파/인스턴스.
  occl : 어절 하나의 전 토큰을 [MASK]로 바꿨을 때의 스팬 확률 하락분 = 그 어절 점수.
         **한 순전파가 전 필드 공용**(마스킹은 필드 무관, 점수만 필드별) - 문서당
         순전파 수 = 창 안 어절 수. 점수는 어절 단위로 직접 산출(재집계 불요).

측정은 기존과 동일: 타당성(IoU@1/5·F1@5·Hit@1, 다른 안 ignore) + 어절 충실도
(COMP/SUFF@k·AOPC, 정답 스팬 보호, [MASK]). 전 표본·3시드. 기존 IG·rollout·random
행은 재계산하지 않고 보고서에서 정본을 복사한다(발주 지시).

사용
    python eval_newattr.py --model ..\runs\seed1\last --out P15_seed1.jsonl
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
    embed_fn = model.get_input_embeddings()

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
            base_logits = model(**inputs).logits
        pred_ids = base_logits.argmax(-1)[0]
        window_widx = sorted({s[0] for s in spans if s is not None})

        # 문서 수준 준비: 필드별 타깃·p_full
        fields = []
        for field, gold in e.gold_spans.items():
            gw = set(gold["word_idx"])
            if not gw:
                continue
            tt = token_idx_for_words(spans, gw)
            if not tt:
                continue
            ll = [int(pred_ids[t]) for t in tt]
            with torch.no_grad():
                pf = float(A.field_score(base_logits, tt, ll))
            fields.append((field, gw, tt, ll, pf))
        if not fields:
            continue

        # ---- occl: 어절당 1순전파, 전 필드 점수 동시 산출 ----
        need_occl = any((e.doc_id, f, "occl") not in done for f, *_ in fields)
        occl_scores = {f: {} for f, *_ in fields}          # field -> {word: drop}
        if need_occl:
            for w in window_widx:
                tk = token_idx_for_words(spans, {w})
                if not tk:
                    continue
                with torch.no_grad():
                    lg = model(**{**inputs, "input_ids": A.mask_tokens(
                        inputs["input_ids"], tk, mask_id)}).logits
                for field, gw, tt, ll, pf in fields:
                    occl_scores[field][w] = pf - float(A.field_score(lg, tt, ll))

        def sc(ov, tt, ll):
            with torch.no_grad():
                return float(A.field_score(model(**{**inputs, **ov}).logits, tt, ll))

        for field, gw, tt, ll, pf in fields:
            special = [t for t, s in enumerate(spans) if s is None]
            gold_tk = token_idx_for_words(spans, gw)

            for method in ("gxi", "occl"):
                if (e.doc_id, field, method) in done:
                    continue
                if method == "gxi":
                    emb = embed_fn(inputs["input_ids"]).detach().clone().requires_grad_(True)
                    kw = {k: v for k, v in inputs.items() if k != "input_ids"}
                    score = A.field_score(model(**kw, inputs_embeds=emb).logits, tt, ll)
                    grad = torch.autograd.grad(score, emb)[0]
                    phi = (grad * emb).sum(-1)[0].abs().detach().float().cpu().tolist()
                    scores, units = aggregate(phi, spans, e.words, "eojeol",
                                              tagger, "all")
                else:
                    pairs = sorted(occl_scores[field].items())
                    units = [{"word_idx": w} for w, _ in pairs]
                    scores = [v for _, v in pairs]
                if not scores:
                    continue

                rec = {"doc_id": e.doc_id, "field": field, "method": method,
                       "level": "eojeol", "mask": "all", "p_full": pf,
                       "plaus": {}, "faith": {}}
                for k in KS:
                    widx = units_to_word_idx(units, scores, k)
                    rec["plaus"][str(k)] = plausibility(
                        widx, sorted(gw), e.raw_boxes,
                        ignore_word_idx=e.ignore_word_idx)
                comps, suffs = [], []
                for k in KS:
                    widx = set(topk_excluding(units, scores, k, gw))
                    tk = token_idx_for_words(spans, widx)
                    if not tk:
                        comps.append(None); suffs.append(None); continue
                    p_wo = sc({"input_ids": A.mask_tokens(
                        inputs["input_ids"], tk, mask_id)}, tt, ll)
                    p_only = sc({"input_ids": A.keep_only(
                        inputs["input_ids"], sorted(set(tk) | set(gold_tk)),
                        mask_id, special)}, tt, ll)
                    cs = comp_suff(pf, p_wo, p_only)
                    comps.append(cs["comp"]); suffs.append(cs["suff"])
                rec["faith"]["comp@k"] = comps
                rec["faith"]["suff@k"] = suffs
                rec["faith"]["aopc_comp"] = aopc(comps)
                emit(rec)

        if (n_doc + 1) % 25 == 0:
            print(f"  {n_doc+1}/{len(ex)} 문서 완료", flush=True)

    f_out.close()
    print(f"완료 → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
