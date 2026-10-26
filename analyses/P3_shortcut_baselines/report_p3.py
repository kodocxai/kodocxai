# -*- coding: utf-8 -*-
"""P3 보고서 - 지름길 기준선 vs 기존 기법 병렬 표.

P3_seed{1,2,3}.jsonl(신규 2종)과 정본 seed{1,2,3}_v2.jsonl(ig·rollout·random,
어절·전체 행만)을 같은 집계 규칙(시드 내 평균 → 시드 간 평균±모표준편차)으로 묶는다.
정본 파일은 읽기만 한다. 출력은 이 폴더 P3_결과_YYYYMMDD.md.

사용: python report_p3.py
"""
from __future__ import annotations

import json
import statistics as st
from collections import defaultdict
from datetime import date
from pathlib import Path

P3 = Path(__file__).resolve().parent
RES = Path(r"D:\AIHub데이터\_exp\results")
SEEDS = (1, 2, 3)
ROWS = ["ig", "rollout", "loc_prior", "label_nearest", "random"]
ROW_KO = {"ig": "IG", "rollout": "rollout", "loc_prior": "위치 prior",
          "label_nearest": "라벨 근접", "random": "무작위(하한)"}
FIELDS = ["제목", "수신", "문서번호", "시행일자", "발신명의"]


def iter_records():
    for s in SEEDS:
        for line in (P3 / f"P3_seed{s}.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                yield s, json.loads(line)
        for line in (RES / f"seed{s}_v2.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("level") == "eojeol" and r.get("mask") == "all":
                yield s, r


def agg(vals_by_seed):
    means = [st.mean(v) for v in vals_by_seed.values() if v]
    ns = [len(v) for v in vals_by_seed.values()]
    if not means:
        return None
    return (st.mean(means), st.pstdev(means) if len(means) > 1 else 0.0, ns)


def fmt(t, d=3):
    if t is None:
        return "-"
    m, s, _ = t
    return f"{m:.{d}f} ± {s:.{d}f}"


def main():
    plaus = defaultdict(lambda: defaultdict(list))   # (method, metric) -> seed -> [v]
    faith = defaultdict(lambda: defaultdict(list))
    fld_plaus = defaultdict(lambda: defaultdict(list))  # (method, field) -> seed -> [iou5]
    n_inst = defaultdict(lambda: defaultdict(set))      # method -> seed -> {(doc,field)}

    for s, r in iter_records():
        m = r["method"]
        n_inst[m][s].add((r["doc_id"], r["field"]))
        p = r.get("plaus", {})
        for k, met in [("1", "iou1"), ("5", "iou5")]:
            v = (p.get(k) or {}).get("iou")
            if v is not None:
                plaus[(m, met)][s].append(v)
        v = (p.get("5") or {}).get("f1")
        if v is not None:
            plaus[(m, "f1_5")][s].append(v)
        v = (p.get("1") or {}).get("hit1")
        if v is not None:
            plaus[(m, "hit1")][s].append(v)
        v5 = (p.get("5") or {}).get("iou")
        if v5 is not None:
            fld_plaus[(m, r["field"])][s].append(v5)
        f = r.get("faith", {})
        ck = f.get("comp@k") or []
        sk = f.get("suff@k") or []
        if len(ck) == 4 and ck[3] is not None:
            faith[(m, "comp10")][s].append(ck[3])
        if len(sk) == 4 and sk[3] is not None:
            faith[(m, "suff10")][s].append(sk[3])
        if f.get("aopc_comp") is not None:
            faith[(m, "aopc")][s].append(f["aopc_comp"])

    L = [f"# P3 지름길 기준선 결과 ({date.today().isoformat()} 자동 생성)", ""]
    L.append("> `report_p3.py` 가 만든다. 손으로 고치지 말 것. 집계 = 시드 내 평균 후")
    L.append("> 시드 간 평균±모표준편차(정본 report_exp.py 와 동일 규칙). 어절 수준·전체 마스킹.")
    L.append("> 신규 2종은 모델 비의존이라 타당성의 시드 편차는 0이 정상이다(충실도만 시드 의존).")
    L.append("")
    L.append("## 표 P3-1. 타당성 - 근거 정답과의 공간 일치 (RQ2 병렬)")
    L.append("")
    L.append("| 기법 | IoU@1 | IoU@5 | F1@5 | Hit@1 | n/시드 |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for m in ROWS:
        ns = sorted({len(v) for v in n_inst[m].values()})
        L.append(f"| {ROW_KO[m]} | {fmt(agg(plaus[(m,'iou1')]))} | {fmt(agg(plaus[(m,'iou5')]))} "
                 f"| {fmt(agg(plaus[(m,'f1_5')]))} | {fmt(agg(plaus[(m,'hit1')]))} | {ns} |")
    L.append("")
    L.append("## 표 P3-2. 충실도 - 텍스트 마스킹 (정답 스팬 보호, RQ1 축 병렬)")
    L.append("")
    L.append("| 기법 | COMP@10 | SUFF@10 | AOPC |")
    L.append("|---|---:|---:|---:|")
    for m in ROWS:
        L.append(f"| {ROW_KO[m]} | {fmt(agg(faith[(m,'comp10')]), 4)} "
                 f"| {fmt(agg(faith[(m,'suff10')]), 4)} | {fmt(agg(faith[(m,'aopc')]), 4)} |")
    L.append("")
    L.append("## 표 P3-3. 필드별 IoU@5 (지름길이 어디서 강한가)")
    L.append("")
    L.append("| 기법 | " + " | ".join(FIELDS) + " |")
    L.append("|---|" + "---:|" * len(FIELDS))
    for m in ROWS:
        cells = [fmt(agg(fld_plaus[(m, f)])) for f in FIELDS]
        L.append(f"| {ROW_KO[m]} | " + " | ".join(cells) + " |")
    L.append("")
    na = {}
    for s in SEEDS:
        p = P3 / f"no_anchor_seed{s}.json"
        if p.exists():
            na[s] = json.loads(p.read_text(encoding="utf-8"))
    if na:
        L.append(f"- label_nearest 앵커 미발견(인스턴스 제외): {json.dumps(na, ensure_ascii=False)}")
        L.append("  (발신명의는 서식 라벨이 없어 정의상 전부 제외 - 표 P3-3 해당 칸은 '-')")
    out = P3 / f"P3_결과_{date.today().strftime('%Y%m%d')}.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"생성: {out}")


if __name__ == "__main__":
    main()
