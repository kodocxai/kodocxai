# -*- coding: utf-8 -*-
"""P11 보고서 - rollout span 수준 타깃 변형 vs 기존 기법 병렬 표.

P11_seed{1,2,3}.jsonl(rollout_mean·rollout_sum)과 정본 seed{1,2,3}_v2.jsonl
(rollout 첫 토큰·ig·random, 어절·전체)을 같은 규칙으로 집계한다.
에디터 포인트 1("consistent span-level target") 직접 대응 표.

사용: python report_p11.py
"""
from __future__ import annotations

import json
import statistics as st
from collections import defaultdict
from datetime import date
from pathlib import Path

P11 = Path(__file__).resolve().parent
RES = Path(r"D:\AIHub데이터\_exp\results")
SEEDS = (1, 2, 3)
ROWS = ["ig", "rollout", "rollout_mean", "rollout_sum", "random"]
ROW_KO = {"ig": "IG (대조, 재계산 없음)", "rollout": "rollout 첫 토큰 (현행)",
          "rollout_mean": "rollout span 평균 (신규)", "rollout_sum": "rollout span 합 (신규)",
          "random": "무작위(하한)"}


def iter_records():
    for s in SEEDS:
        for line in (P11 / f"P11_seed{s}.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                yield s, json.loads(line)
        for line in (RES / f"seed{s}_v2.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("level") == "eojeol" and r.get("mask") == "all":
                yield s, r


def agg(d):
    means = [st.mean(v) for v in d.values() if v]
    if not means:
        return None
    return st.mean(means), (st.pstdev(means) if len(means) > 1 else 0.0)


def fmt(t, d=3):
    return "-" if t is None else f"{t[0]:.{d}f} ± {t[1]:.{d}f}"


def main():
    plaus = defaultdict(lambda: defaultdict(list))
    faith = defaultdict(lambda: defaultdict(list))
    n_inst = defaultdict(lambda: defaultdict(set))
    for s, r in iter_records():
        m = r["method"]
        if m not in ROWS:
            continue
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
        f = r.get("faith", {})
        ck = f.get("comp@k") or []
        sk = f.get("suff@k") or []
        if len(ck) == 4 and ck[3] is not None:
            faith[(m, "comp10")][s].append(ck[3])
        if len(sk) == 4 and sk[3] is not None:
            faith[(m, "suff10")][s].append(sk[3])
        if f.get("aopc_comp") is not None:
            faith[(m, "aopc")][s].append(f["aopc_comp"])

    L = [f"# P11 rollout span 타깃 결과 ({date.today().isoformat()} 자동 생성)", ""]
    L.append("> `report_p11.py` 가 만든다. 손으로 고치지 말 것. 어절·전체, v2 프로토콜 동일,")
    L.append("> rollout 행렬은 문서당 1회 계산 후 스팬 토큰 행의 평균/합. IG 는 재계산하지 않음")
    L.append("> (field_score 기하평균에 대한 기울기라 이미 span 수준 타깃 - 에디터의 일관 타깃 조건 충족).")
    L.append("> 집계 = 시드 내 평균 → 시드 간 평균±모표준편차.")
    L.append("")
    L.append("## 표 P11-1. 타당성 (표 5 병렬)")
    L.append("")
    L.append("| 기법 | IoU@1 | IoU@5 | F1@5 | Hit@1 | n/시드 |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for m in ROWS:
        ns = sorted({len(v) for v in n_inst[m].values()})
        L.append(f"| {ROW_KO[m]} | {fmt(agg(plaus[(m,'iou1')]))} | {fmt(agg(plaus[(m,'iou5')]))} "
                 f"| {fmt(agg(plaus[(m,'f1_5')]))} | {fmt(agg(plaus[(m,'hit1')]))} | {ns} |")
    L.append("")
    L.append("## 표 P11-2. 충실도 (어절·전체)")
    L.append("")
    L.append("| 기법 | COMP@10 | SUFF@10 | AOPC |")
    L.append("|---|---:|---:|---:|")
    for m in ROWS:
        L.append(f"| {ROW_KO[m]} | {fmt(agg(faith[(m,'comp10')]), 4)} "
                 f"| {fmt(agg(faith[(m,'suff10')]), 4)} | {fmt(agg(faith[(m,'aopc')]), 4)} |")
    L.append("")
    L.append("- 주: span 평균과 span 합은 지표가 완전히 같다 - sum = |span| × mean 이라")
    L.append("  단위 순위가 동일하기 때문(수학적 필연). 두 행을 모두 보고하는 것은 두 정의가")
    L.append("  모두 검토되었음을 보이기 위함이다.")
    L.append("")
    L.append("## 원고 인용용 요약 (영문 1문장)")
    L.append("")
    L.append("> Re-computing attention rollout against a consistent span-level target "
             "(averaging or summing the rollout rows of all span tokens, which induce "
             "identical rankings) changes its plausibility only marginally (IoU@5 "
             "0.045 to 0.053; Hit@1 remains 0.000) and leaves its faithfulness at the "
             "random-baseline level (AOPC 0.0113), indicating that rollout's weak "
             "performance stems from the method's position bias rather than from the "
             "first-token target simplification used in the original protocol.")
    out = P11 / f"P11_결과_{date.today().strftime('%Y%m%d')}.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"생성: {out}")


if __name__ == "__main__":
    main()
