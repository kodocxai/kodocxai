# -*- coding: utf-8 -*-
"""P3 지름길 기준선 2종 - field-location prior · label-nearest (검토서 §2.3).

정형 서식에서는 「제목은 여기, 발신명의는 아래」라는 위치 지식만으로 높은 공간
점수를 얻을 수 있다. 기존 random 은 「무작위보다 낫다」만 보여주므로, 비학습
지름길 2종을 추가해 IG·rollout 과 전 지표에서 병렬 비교한다(보강 패키지 P3).

기준선 정의 (둘 다 모델 비의존·비학습·결정적)
  loc_prior     : 훈련 분할(약지도 5,000건)에서 필드별 값 스팬 중심의 평균 위치
                  (0~1000 정규화)를 구하고, 평가 문서의 각 어절을 그 평균점과의
                  유클리드 거리 역순으로 순위 매긴다. 시험 문서 주석은 쓰지 않는다.
  label_nearest : 평가 문서에서 필드 라벨 앵커(「제 목」·「수 신」 등, 어절 분리
                  대응)를 찾아 앵커 중심과의 거리 역순으로 순위 매긴다. 앵커 어절
                  자신은 후보에서 제외. 앵커가 없는 필드(발신명의는 서식 라벨이
                  없다)는 그 인스턴스를 건너뛰고 커버리지를 보고한다.

측정 프로토콜은 v2 와 동일 - 후보는 512 창 안 어절, 정답 스팬 보호(topk_excluding),
[MASK] 치환, p_full = 예측 라벨 기하평균(kodx_attrib.field_score), COMP/SUFF@k(1·3·5·10),
AOPC = k 격자 COMP 평균(aopc_comp). 타당성은 plausibility(IoU@k·F1·hit@1, 다른 안 ignore).
모달 교란·NAOPC 는 범위 밖(P3 는 귀속 순위 기준선만 추가한다).

사용
    python eval_shortcut.py --model ..\runs\seed1\last --seed 1 --out P3_seed1.jsonl
결과는 이 폴더에만 쓴다. frozen 자산·정본 결과 무접촉.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch

EXP = Path(r"D:\AIHub데이터\_exp")
P3 = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))

from kodx_data import build_examples, load_processor                 # noqa: E402
from kodx_units import subword_char_spans                            # noqa: E402
from kodx_metrics import aopc, comp_suff, plausibility               # noqa: E402
import kodx_attrib as A                                              # noqa: E402
from kodx_eval import token_idx_for_words, topk_excluding            # noqa: E402

FIELDS = ["제목", "수신", "문서번호", "시행일자", "발신명의"]
KS = [1, 3, 5, 10]
METHODS = ["loc_prior", "label_nearest"]

# 필드 라벨 앵커 - 단일 어절 표기와 어절 분리 표기(「제 목」) 둘 다 잡는다.
# 구두점·콜론이 붙은 표기(「제목:」)는 스트립 후 비교한다.
ANCHORS = {
    "제목": [("제목",), ("제", "목")],
    "수신": [("수신",), ("수", "신")],
    "문서번호": [("문서번호",), ("문서", "번호")],
    "시행일자": [("시행일자",), ("시행", "일자")],
    # 발신명의: 서식 라벨이 없다 - label_nearest 정의 불가, 건너뛴다
}
_STRIP = re.compile(r"[^\w가-힣]")


def norm_word(t: str) -> str:
    return _STRIP.sub("", t)


def build_prior(train_jsonl: Path, cache: Path) -> dict:
    """훈련 분할 약지도 라벨에서 필드별 값 스팬 중심의 평균 위치(0~1000)를 구한다."""
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    acc = defaultdict(lambda: [0.0, 0.0, 0])
    with train_jsonl.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            W, H = d["image"]["width"], d["image"]["height"]
            if not W or not H:
                continue
            for fld, info in (d.get("fields") or {}).items():
                bb = info.get("bbox")
                if fld not in FIELDS or not bb:
                    continue
                x, y, w, h = bb
                cx = (x + w / 2) / W * 1000
                cy = (y + h / 2) / H * 1000
                a = acc[fld]
                a[0] += cx; a[1] += cy; a[2] += 1
    prior = {}
    for fld, (sx, sy, n) in acc.items():
        prior[fld] = {"cx": sx / n, "cy": sy / n, "n": n}
    cache.write_text(json.dumps(prior, ensure_ascii=False, indent=1), encoding="utf-8")
    return prior


def find_anchor(words: list[str], boxes: list[list[int]], field: str):
    """필드 라벨 앵커의 (어절 인덱스 목록, 중심). 최상단(작은 y) 매치를 쓴다."""
    pats = ANCHORS.get(field)
    if not pats:
        return None
    cands = []
    toks = [norm_word(w) for w in words]
    for pat in pats:
        L = len(pat)
        for i in range(len(toks) - L + 1):
            if tuple(toks[i:i + L]) == pat:
                idxs = list(range(i, i + L))
                cy = sum((boxes[j][1] + boxes[j][3]) / 2 for j in idxs) / L
                cx = sum((boxes[j][0] + boxes[j][2]) / 2 for j in idxs) / L
                cands.append((cy, cx, idxs))
    if not cands:
        return None
    cy, cx, idxs = min(cands)          # 최상단 우선, 동률이면 좌측
    return idxs, (cx, cy)


def center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def main() -> int:
    ap = argparse.ArgumentParser(description="P3 지름길 기준선 평가")
    ap.add_argument("--model", required=True)
    ap.add_argument("--docs", default=str(Path(r"D:\AIHub데이터\_pilot\docs.jsonl")))
    ap.add_argument("--images", default=str(EXP / "data" / "pilot_images224"))
    ap.add_argument("--annotations",
                    default=str(Path(r"D:\AIHub데이터\_pilot\annotations_park.jsonl")))
    ap.add_argument("--train", default=str(EXP / "data" / "train.jsonl"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    from transformers import AutoModelForTokenClassification
    from PIL import Image

    prior = build_prior(Path(a.train), P3 / "prior_train.json")
    print("위치 prior:", {k: (round(v["cx"]), round(v["cy"]), v["n"]) for k, v in prior.items()})

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = load_processor(a.model, apply_ocr=False)
    model = AutoModelForTokenClassification.from_pretrained(a.model).to(dev).eval()
    tok = proc.tokenizer
    mask_id = tok.mask_token_id

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
        print(f"이미 계산된 조합 {len(done)}개 - 건너뛴다")
    f_out = out.open("a", encoding="utf-8")

    def emit(rec):
        f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f_out.flush(); os.fsync(f_out.fileno())

    no_anchor = defaultdict(int)

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

        # 후보 = 512 창 안에 실제로 든 어절 (다른 기법과 같은 후보 집합)
        window_widx = sorted({s[0] for s in spans if s is not None})

        def sc(ov=None):
            inp = dict(inputs) if ov is None else {**inputs, **ov}
            lg = model(**inp).logits
            return A.field_score(lg, tgt_tokens, label_ids)

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

            for method in METHODS:
                key = (e.doc_id, field, method)
                if key in done:
                    continue
                if method == "loc_prior":
                    pr = prior.get(field)
                    if pr is None:
                        continue
                    ref = (pr["cx"], pr["cy"])
                    excl = set()
                else:
                    an = find_anchor(e.words, e.boxes, field)
                    if an is None:
                        no_anchor[field] += 1
                        continue
                    anchor_idxs, ref = an
                    excl = set(anchor_idxs)

                cand = [w for w in window_widx if w not in excl]
                scores = [-math.dist(center(e.boxes[w]), ref) for w in cand]
                units = [{"word_idx": w} for w in cand]

                rec = {"doc_id": e.doc_id, "field": field, "method": method,
                       "level": "eojeol", "mask": "all", "p_full": p_full,
                       "plaus": {}, "faith": {}}

                ranked = sorted(range(len(cand)), key=lambda i: -scores[i])
                for k in KS:
                    widx = [cand[i] for i in ranked[:k]]
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

        if (n_doc + 1) % 25 == 0:
            print(f"  {n_doc+1}/{len(ex)} 문서 완료", flush=True)

    f_out.close()
    if no_anchor:
        print("앵커 미발견(인스턴스 건너뜀):", dict(no_anchor))
    (P3 / f"no_anchor_seed{a.seed}.json").write_text(
        json.dumps(dict(no_anchor), ensure_ascii=False), encoding="utf-8")
    print(f"완료 → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
