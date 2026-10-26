# -*- coding: utf-8 -*-
"""P5 보고서 - 설명 대상 분리 재집계 + error-grounding (검토서 §2.7).

분류(시드·문서·필드, 정답 실재·512 창 안):
  correct       예측 스팬 = 정답 스팬 (완전일치)
  wrong-partial 예측이 있고 정답과 일부만 겹침
  wrong-disjoint예측이 있고 정답과 전혀 안 겹침
  missed        예측 없음(모델이 O 처리)
  (spurious     정답 absent 인데 예측 있음 - 귀속 레코드가 없어 빈도만 보고)

주 분석 = correct 전용 재집계(타당성·충실도, v2 레코드·어절·전체).
error-grounding = wrong 군에서 귀속 top-k 를 「정답」이 아니라 「모델의 예측 스팬」과
대조한다 - 귀속이 모델이 실제로 본 곳을 가리키면 설명은 충실히 작동한 것이다.

8월 정확예측_조건부.md(2분법)와의 정합: correct = 그 문서의 「완전일치 맞힘」과 동일
기준이므로 군 크기가 일치해야 한다(검증 출력).

사용: python report_p5.py
"""
from __future__ import annotations

import json
import statistics as st
from collections import defaultdict
from datetime import date
from pathlib import Path
import sys

EXP = Path(r"D:\AIHub데이터\_exp")
P5 = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))
from kodx_data import build_examples                                 # noqa: E402
from kodx_metrics import plausibility                                # noqa: E402

SEEDS = (1, 2, 3)
METHODS = ["ig", "rollout", "random"]
METHOD_KO = {"ig": "IG", "rollout": "rollout", "random": "무작위(하한)"}
GROUPS = ["correct", "wrong-partial", "wrong-disjoint", "missed"]


def agg(by_seed):
    means = [st.mean(v) for v in by_seed.values() if v]
    if not means:
        return None
    return st.mean(means), (st.pstdev(means) if len(means) > 1 else 0.0)


def fmt(t, d=3):
    return "-" if t is None else f"{t[0]:.{d}f} ± {t[1]:.{d}f}"


def main():
    # 문서 메타(raw_boxes·ignore) - 이미지 불필요
    ex = build_examples(Path(r"D:\AIHub데이터\_pilot\docs.jsonl"), None,
                        Path(r"D:\AIHub데이터\_pilot\annotations_park.jsonl"),
                        require_image=False)
    meta = {e.doc_id: e for e in ex}

    cls = {}                      # (seed, doc, field) -> group
    pred_span = {}                # (seed, doc, field) -> pred 어절 목록
    topk = {}                     # (seed, doc, field, method) -> top10
    spurious = defaultdict(lambda: defaultdict(int))   # seed -> field -> n
    for s in SEEDS:
        for line in (P5 / f"P5_pred_seed{s}.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r["type"] == "topk":
                topk[(s, r["doc_id"], r["field"], r["method"])] = r["top10"]
                continue
            g, p = set(r["gold"]), set(r["pred"])
            if not r["gold"]:
                if p:
                    spurious[s][r["field"]] += 1
                continue
            if not r["in_window"]:
                continue
            key = (s, r["doc_id"], r["field"])
            pred_span[key] = sorted(p)
            if p == g:
                cls[key] = "correct"
            elif not p:
                cls[key] = "missed"
            elif p & g:
                cls[key] = "wrong-partial"
            else:
                cls[key] = "wrong-disjoint"

    n_by = defaultdict(lambda: defaultdict(int))
    for (s, _, _), grp in cls.items():
        n_by[grp][s] += 1
    print("군 크기(시드별):", {g: dict(n_by[g]) for g in GROUPS})
    print("spurious(시드별 합):", {s: sum(d.values()) for s, d in spurious.items()})

    # v2 레코드(어절·전체) 조인
    plaus = defaultdict(lambda: defaultdict(list))   # (method, group, metric)
    faith = defaultdict(lambda: defaultdict(list))
    for s in SEEDS:
        for line in (EXP / "results" / f"seed{s}_v2.jsonl").read_text(
                encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("level") != "eojeol" or r.get("mask") != "all":
                continue
            grp = cls.get((s, r["doc_id"], r["field"]))
            if grp is None:
                continue
            m = r["method"]
            p5v = (r["plaus"].get("5") or {})
            p1v = (r["plaus"].get("1") or {})
            if p5v.get("iou") is not None:
                plaus[(m, grp, "iou5")][s].append(p5v["iou"])
            if p1v.get("hit1") is not None:
                plaus[(m, grp, "hit1")][s].append(p1v["hit1"])
            ck = (r["faith"].get("comp@k") or [None] * 4)
            if len(ck) == 4 and ck[3] is not None:
                faith[(m, grp, "comp10")][s].append(ck[3])
            if r["faith"].get("aopc_comp") is not None:
                faith[(m, grp, "aopc")][s].append(r["faith"]["aopc_comp"])

    # error-grounding: wrong 군에서 top-k vs 예측 스팬
    eg = defaultdict(lambda: defaultdict(list))      # (method, group, metric)
    for (s, doc, fld), grp in cls.items():
        if grp not in ("wrong-partial", "wrong-disjoint"):
            continue
        e = meta.get(doc)
        ps = pred_span.get((s, doc, fld))
        if e is None or not ps:
            continue
        for m in METHODS:
            t10 = topk.get((s, doc, fld, m))
            if not t10:
                continue
            r5 = plausibility(t10[:5], ps, e.raw_boxes,
                              ignore_word_idx=e.ignore_word_idx)
            r1 = plausibility(t10[:1], ps, e.raw_boxes,
                              ignore_word_idx=e.ignore_word_idx)
            if r5.get("iou") is not None:
                eg[(m, grp, "iou5")][s].append(r5["iou"])
            if r1.get("hit1") is not None:
                eg[(m, grp, "hit1")][s].append(r1["hit1"])

    L = [f"# P5 설명 대상 분리 결과 ({date.today().isoformat()} 자동 생성)", ""]
    L.append("> `report_p5.py` 가 만든다. 손으로 고치지 말 것. 어절·전체, v2 레코드 재집계,")
    L.append("> 집계 = 시드 내 평균 → 시드 간 평균±모표준편차. 분류는 P5_pred_seed{N}.jsonl")
    L.append("> (eval_kie.decode_spans 동일 디코딩). 8월 정확예측_조건부.md 의 2분법을")
    L.append("> 4군으로 확장한 것 - correct 군 크기는 그 문서의 완전일치 848(시드 평균)과 정합해야 한다.")
    L.append("")
    ns = {g: round(st.mean(list(n_by[g].values())), 1) if n_by[g] else 0 for g in GROUPS}
    L.append(f"- 군 크기(시드 평균): {json.dumps(ns, ensure_ascii=False)} / 총 1,166")
    fb = defaultdict(lambda: defaultdict(int))
    for (s, doc, fld), grp in cls.items():
        fb[grp][fld] += 1
    for g in ("missed", "wrong-partial"):
        d = {f: round(v / len(SEEDS), 1) for f, v in
             sorted(fb[g].items(), key=lambda kv: -kv[1])}
        L.append(f"- {g} 필드별(시드 평균): {json.dumps(d, ensure_ascii=False)}")
    sp = {f: round(st.mean([spurious[s].get(f, 0) for s in SEEDS]), 1)
          for f in sorted({f for d in spurious.values() for f in d})}
    L.append(f"- **spurious(absent 인데 예측함, 시드 평균)**: {json.dumps(sp, ensure_ascii=False)}")
    L.append("  (absent 필드에는 귀속 레코드가 없어 빈도만 보고 - 추가 귀속 산출은 확장 항목)")
    L.append("")
    L.append("## 표 P5-1. 군별 타당성 (vs 정답 스팬)")
    L.append("")
    L.append("| 기법 | 지표 | " + " | ".join(GROUPS) + " |")
    L.append("|---|---|" + "---:|" * len(GROUPS))
    for m in METHODS:
        for met, ko in [("iou5", "IoU@5"), ("hit1", "Hit@1")]:
            cells = [fmt(agg(plaus[(m, g, met)])) for g in GROUPS]
            L.append(f"| {METHOD_KO[m]} | {ko} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("## 표 P5-2. 군별 충실도")
    L.append("")
    L.append("| 기법 | 지표 | " + " | ".join(GROUPS) + " |")
    L.append("|---|---|" + "---:|" * len(GROUPS))
    for m in METHODS:
        for met, ko in [("comp10", "COMP@10"), ("aopc", "AOPC")]:
            cells = [fmt(agg(faith[(m, g, met)], ), 4) for g in GROUPS]
            L.append(f"| {METHOD_KO[m]} | {ko} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("## 표 P5-3. error-grounding - wrong 군에서 귀속 vs **모델의 예측 스팬**")
    L.append("")
    L.append("귀속 대상은 모델의 실제 예측이므로, 틀린 예측에서 귀속이 예측 스팬을 짚으면")
    L.append("설명은 충실히 작동한 것이다(낮은 vs-정답 타당성은 「모델이 딴 곳을 봄」의 증거).")
    L.append("")
    L.append("| 기법 | 지표 | wrong-partial | wrong-disjoint |")
    L.append("|---|---|---:|---:|")
    for m in METHODS:
        for met, ko in [("iou5", "IoU@5(vs예측)"), ("hit1", "Hit@1(vs예측)")]:
            cells = [fmt(agg(eg[(m, g, met)])) for g in ("wrong-partial", "wrong-disjoint")]
            L.append(f"| {METHOD_KO[m]} | {ko} | " + " | ".join(cells) + " |")
    out = P5 / f"P5_결과_{date.today().strftime('%Y%m%d')}.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"생성: {out}")


if __name__ == "__main__":
    main()
