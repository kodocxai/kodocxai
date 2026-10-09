"""필드별 KIE 성능 (P/R/F1) — 외부 검토 P3 대응.

외부 검토 3곳이 공통으로 지적했다: **설명을 평가하기 전에 모델이 쓸 만한지부터
보여야 한다.** 예측이 틀린 상태에서 설명 지표가 낮은 것과, 맞은 예측의 설명이
부적절한 것은 다른 이야기인데 지금 원고로는 구분할 수 없다.

기존 `seedN_v2.jsonl` 에는 이 값이 없다 — 귀속·교란 기록이라 예측 정확도를 담지 않는다.
그래서 새로 산출한다. **학습이 아니라 추론만** 하므로 시드당 수 분이면 된다.

채점 방식
    스팬 단위 정확 일치. 예측 BIO 를 필드별 스팬으로 디코딩해 정답 어절 집합과
    **완전히 같을 때만** 맞은 것으로 센다(부분 일치는 오답).
    어절 단위 P/R/F1 도 함께 낸다 — 부분 일치를 반영한 느슨한 기준이다.

    평가 대상은 **정답 스팬이 있는 (문서, 필드)** 다(`absent` 제외). 512 토큰 창을
    벗어난 정답은 예측이 원리적으로 불가능하므로 별도로 집계해 보고한다.

사용
    python eval_kie.py --model runs/seed1/last --out results/kie_seed1.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image

from kodx_data import ID2LABEL, build_examples, load_processor
from kodx_units import subword_char_spans

FIELDS = ["제목", "수신", "문서번호", "시행일자", "발신명의"]


def decode_spans(pred_ids, spans, n_words: int) -> dict:
    """토큰 BIO 예측 → 필드별 어절 집합.

    한 어절에 여러 서브워드가 걸리므로 **다수결**로 어절 라벨을 정한다.
    B- 와 I- 는 필드만 같으면 같은 스팬으로 본다(경계 판정은 채점 대상이 아니다).
    """
    votes = [defaultdict(int) for _ in range(n_words)]
    for t, s in enumerate(spans):
        if s is None or t >= len(pred_ids):
            continue
        lab = ID2LABEL.get(int(pred_ids[t]), "O")
        if lab == "O":
            votes[s[0]]["O"] += 1
        else:
            votes[s[0]][lab.split("-", 1)[1]] += 1

    out = defaultdict(set)
    for w, v in enumerate(votes):
        if not v:
            continue
        best = max(v, key=v.get)
        if best != "O":
            out[best].add(w)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--docs", default=r"D:\AIHub데이터\_pilot\docs.jsonl")
    ap.add_argument("--images", default=r"D:\AIHub데이터\_pilot\images")
    ap.add_argument("--annotations", default=r"D:\AIHub데이터\_pilot\annotations_park.jsonl")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from transformers import AutoModelForTokenClassification

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = load_processor(a.model, apply_ocr=False)
    tok = proc.tokenizer
    model = AutoModelForTokenClassification.from_pretrained(a.model).to(dev).eval()

    ex = build_examples(Path(a.docs), Path(a.images), Path(a.annotations))
    print(f"평가 문서 {len(ex)}건 / 장치 {dev}")

    # 필드별 집계
    tp_w = defaultdict(int); fp_w = defaultdict(int); fn_w = defaultdict(int)
    exact = defaultdict(int); total = defaultdict(int); oob = defaultdict(int)
    # 인스턴스별 정오 딱지. 표 3·4 를 「정확 예측 조건부」로 다시 가르는 데 쓴다.
    # 합계만 저장하면 이 분해를 못 하므로 (문서, 필드)별로 남긴다.
    inst: list[dict] = []

    for n, e in enumerate(ex):
        if not e.gold_spans:
            continue
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
            gw = set(g["word_idx"]) if g else set()
            if not gw:
                continue
            total[f] += 1
            if not (gw & covered):
                oob[f] += 1          # 512 창 밖 - 예측이 원리적으로 불가능
                continue
            pw = pred.get(f, set())
            tp_w[f] += len(pw & gw)
            fp_w[f] += len(pw - gw)
            fn_w[f] += len(gw - pw)
            em = int(pw == gw)
            exact[f] += em
            inter = len(pw & gw)
            inst.append({
                "doc_id": e.doc_id, "field": f,
                "exact": em,                                   # 스팬 완전일치
                "hit": int(inter > 0),                          # 부분이라도 맞음
                "n_gold": len(gw), "n_pred": len(pw), "n_inter": inter,
            })

        if (n + 1) % 50 == 0:
            print(f"  {n+1}/{len(ex)}", flush=True)

    res = {"model": a.model, "fields": {}, "instances": inst}
    for f in FIELDS:
        p = tp_w[f] / (tp_w[f] + fp_w[f]) if (tp_w[f] + fp_w[f]) else 0.0
        r = tp_w[f] / (tp_w[f] + fn_w[f]) if (tp_w[f] + fn_w[f]) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        evaluable = total[f] - oob[f]
        res["fields"][f] = {
            "precision": p, "recall": r, "f1": f1,
            "exact_match": exact[f] / evaluable if evaluable else 0.0,
            "instances": total[f], "out_of_window": oob[f], "evaluable": evaluable,
        }
        print(f"  {f:5s} P {p:.3f}  R {r:.3f}  F1 {f1:.3f}  "
              f"EM {res['fields'][f]['exact_match']:.3f}  "
              f"(n={evaluable}, 창밖 {oob[f]})")

    # 마이크로 평균
    TP, FP, FN = sum(tp_w.values()), sum(fp_w.values()), sum(fn_w.values())
    p = TP / (TP + FP) if (TP + FP) else 0.0
    r = TP / (TP + FN) if (TP + FN) else 0.0
    res["micro"] = {"precision": p, "recall": r,
                    "f1": 2 * p * r / (p + r) if (p + r) else 0.0}
    print(f"  micro P {p:.3f}  R {r:.3f}  F1 {res['micro']['f1']:.3f}")

    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"저장: {a.out}")


if __name__ == "__main__":
    main()
