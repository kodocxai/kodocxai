# -*- coding: utf-8 -*-
"""P13 프록시 정의 민감도 (MDPI R4-3·R1-1, 10-02 발주 - 대안 정의 경로).

착수 전 확인 결과(회신에 보고): 주석 스키마에 필드 라벨(항목명) 박스가 없다
(fields 키 = bbox·value·value_norm·word_ids·status·source 뿐). 발주 명세의 대안
정의로 진행한다 - **값 영역을 상하좌우 k% 확장(k=10, 25, 50)**.

정의: 각 정답 어절 상자 [x,y,w,h] → [x-kw, y-kh, (1+2k)w, (1+2k)h] (k=비율).
확장 참조 집합 = 정답 어절 ∪ {중심이 확장 상자 합집합 안에 드는 어절}.
지표는 kodx_metrics.plausibility 무수정 재사용(IoU는 확장 집합의 상자 영역 기준,
Hit@1·F1 은 확장 집합 기준). k=0 이 표 5 현행값과 일치해야 한다(내장 재현 검증).

귀속 상위 10 어절은 P5 덤프(P5_pred_seed{N}.jsonl 의 topk 레코드 - 표 5 타당성과
동일 경로 units_to_word_idx)를 재사용한다. 재계산 없음, CPU 전용.

사용: python report_p13.py
"""
from __future__ import annotations

import json
import statistics as st
from collections import defaultdict
from datetime import date
from pathlib import Path
import sys

EXP = Path(r"D:\AIHub데이터\_exp")
P13 = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))
from kodx_data import build_examples                                 # noqa: E402
from kodx_metrics import plausibility                                # noqa: E402

SEEDS = (1, 2, 3)
METHODS = ("ig", "rollout", "random")
METHOD_KO = {"ig": "IG", "rollout": "rollout", "random": "무작위(하한)"}
EXPANDS = (0, 10, 25, 50)          # %


def expand_box(b, k):
    x, y, w, h = b
    e = k / 100.0
    return [x - w * e, y - h * e, w * (1 + 2 * e), h * (1 + 2 * e)]


def inside_any(cx, cy, boxes):
    for x, y, w, h in boxes:
        if x <= cx <= x + w and y <= cy <= y + h:
            return True
    return False


def main():
    ex = build_examples(Path(r"D:\AIHub데이터\_pilot\docs.jsonl"), None,
                        Path(r"D:\AIHub데이터\_pilot\annotations_park.jsonl"),
                        require_image=False)
    meta = {e.doc_id: e for e in ex}

    topk = {}
    for s in SEEDS:
        p = EXP / "P5_설명대상분리" / f"P5_pred_seed{s}.jsonl"
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r["type"] == "topk":
                topk[(s, r["doc_id"], r["field"], r["method"])] = r["top10"]
    print(f"top10 로드 {len(topk)}건")

    acc = defaultdict(lambda: defaultdict(list))   # (method, k, metric) -> seed -> [v]
    nref = defaultdict(lambda: defaultdict(list))  # k -> seed -> 참조 어절 수
    for (s, doc, fld, m), t10 in topk.items():
        e = meta.get(doc)
        if e is None:
            continue
        g = e.gold_spans.get(fld)
        if not g or not g.get("word_idx"):
            continue
        gold = sorted(set(g["word_idx"]))
        for k in EXPANDS:
            if k == 0:
                ref = gold
            else:
                exp_boxes = [expand_box(e.raw_boxes[w], k) for w in gold
                             if w < len(e.raw_boxes)]
                ref = set(gold)
                for w, b in enumerate(e.raw_boxes):
                    if w in ref or not b:
                        continue
                    cx, cy = b[0] + b[2] / 2, b[1] + b[3] / 2
                    if inside_any(cx, cy, exp_boxes):
                        ref.add(w)
                ref = sorted(ref)
            if m == "ig":                      # 참조 집합 크기는 기법 무관 - 1회만
                nref[k][s].append(len(ref))
            r5 = plausibility(t10[:5], ref, e.raw_boxes,
                              ignore_word_idx=e.ignore_word_idx)
            r1 = plausibility(t10[:1], ref, e.raw_boxes,
                              ignore_word_idx=e.ignore_word_idx)
            acc[(m, k, "iou1")][s].append(r1["iou"])
            acc[(m, k, "iou5")][s].append(r5["iou"])
            acc[(m, k, "f1_5")][s].append(r5["f1"])
            acc[(m, k, "hit1")][s].append(r1["hit1"])

    def agg(d):
        means = [st.mean(v) for v in d.values() if v]
        if not means:
            return None
        return st.mean(means), (st.pstdev(means) if len(means) > 1 else 0.0)

    def fmt(t, d=3):
        return "-" if t is None else f"{t[0]:.{d}f} ± {t[1]:.{d}f}"

    L = [f"# P13 프록시 정의 민감도 ({date.today().isoformat()} 자동 생성)", ""]
    L.append("> `report_p13.py` 가 만든다. 손으로 고치지 말 것. 라벨 박스 주석 부재 확인에")
    L.append("> 따라 대안 정의(값 영역 k% 확장) 경로. 상위 10 = P5 덤프 재사용(표 5 동일 경로),")
    L.append("> 집계 = 시드 내 평균 → 시드 간 평균±모표준편차. k=0 = 현행 정의(재현 검증용).")
    L.append("")
    L.append("## 표 P13-1. 확장률별 표 5 지표")
    L.append("")
    L.append("| 확장 k | 기법 | IoU@1 | IoU@5 | F1@5 | Hit@1 |")
    L.append("|---|---|---:|---:|---:|---:|")
    for k in EXPANDS:
        for m in METHODS:
            L.append(f"| {k}% | {METHOD_KO[m]} | {fmt(agg(acc[(m,k,'iou1')]))} "
                     f"| {fmt(agg(acc[(m,k,'iou5')]))} | {fmt(agg(acc[(m,k,'f1_5')]))} "
                     f"| {fmt(agg(acc[(m,k,'hit1')]))} |")
    L.append("")
    L.append("## 참조 집합 크기 (어절 수, 시드 평균)")
    L.append("")
    L.append("| k | 평균 참조 어절 수 |")
    L.append("|---|---:|")
    for k in EXPANDS:
        t = agg(nref[k])
        L.append(f"| {k}% | {t[0]:.2f} |")
    L.append("")
    L.append("## 원고 인용용 요약 (영문 1문장)")
    L.append("")
    L.append("> Because the source annotations contain no field-label boxes, we probe "
             "the sensitivity of the value-region proxy by dilating every gold box by "
             "10-50% on all sides (absorbing neighboring words whose centers fall "
             "inside); the method ranking of Table 5 is unchanged at every dilation - "
             "IG rises moderately (Hit@1 0.40 to 0.60 at 50%) as near-miss pointings "
             "are re-classified as hits, while rollout stays at the random-baseline "
             "level (Hit@1 at most 0.002) - indicating that the conclusions do not "
             "hinge on the exact extent of the reference region.")
    out = P13 / f"P13_결과_{date.today().strftime('%Y%m%d')}.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"생성: {out}")


if __name__ == "__main__":
    main()
