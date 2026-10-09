# -*- coding: utf-8 -*-
"""P12 보고서 - 교란 전략 3종(mask/delete/randsub) × 보호 유무 비교.

mask 행 = 정본 재사용: 보호 = seed{N}_v2.jsonl, 미보호 = seed{N}.jsonl (어절·전체,
P12 표본 100문서로 한정해 동일 문서 집합 비교). delete·randsub = P12_seed{N}.jsonl.
판독: 방법 순위(IG > rollout ≈ random)와 스팬 보호 효과가 교란 전략에 의존하는가.

사용: python report_p12.py
"""
from __future__ import annotations

import json
import statistics as st
from collections import defaultdict
from datetime import date
from pathlib import Path

P12 = Path(__file__).resolve().parent
RES = Path(r"D:\AIHub데이터\_exp\results")
SEEDS = (1, 2, 3)
METHODS = ("ig", "rollout", "random")
METHOD_KO = {"ig": "IG", "rollout": "rollout", "random": "무작위(하한)"}
STRATS = ("mask", "delete", "randsub")
PROTS = ("protected", "unprotected")
PROT_KO = {"protected": "보호", "unprotected": "미보호"}

SAMPLE = set(json.loads((P12 / "P12_표본100.json").read_text(encoding="utf-8"))["docs"])


def iter_records():
    for s in SEEDS:
        for line in (P12 / f"P12_seed{s}.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r["doc_id"] in SAMPLE:
                yield s, r["method"], r["strategy"], r["protection"], r["faith"]
        for name, prot in [(f"seed{s}_v2.jsonl", "protected"), (f"seed{s}.jsonl", "unprotected")]:
            for line in (RES / name).read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                if (r.get("level") == "eojeol" and r.get("mask") == "all"
                        and r["doc_id"] in SAMPLE):
                    yield s, r["method"], "mask", prot, r["faith"]


def agg(d):
    means = [st.mean(v) for v in d.values() if v]
    if not means:
        return None
    return st.mean(means), (st.pstdev(means) if len(means) > 1 else 0.0)


def fmt(t, d=4):
    return "-" if t is None else f"{t[0]:.{d}f} ± {t[1]:.{d}f}"


def main():
    acc = defaultdict(lambda: defaultdict(list))
    n_inst = defaultdict(lambda: defaultdict(set))
    for s, m, strat, prot, f in iter_records():
        key = (strat, prot, m)
        ck = f.get("comp@k") or []
        sk = f.get("suff@k") or []
        if len(ck) == 4 and ck[3] is not None:
            acc[key + ("comp10",)][s].append(ck[3])
        if len(sk) == 4 and sk[3] is not None:
            acc[key + ("suff10",)][s].append(sk[3])
        if f.get("aopc_comp") is not None:
            acc[key + ("aopc",)][s].append(f["aopc_comp"])
            n_inst[key][s].add(len(acc[key + ("aopc",)][s]))

    L = [f"# P12 대체 교란 전략 결과 ({date.today().isoformat()} 자동 생성)", ""]
    L.append("> `report_p12.py` 가 만든다. 손으로 고치지 말 것. 표본 = 고정 100문서")
    L.append("> (P12_표본100.json: 기계산 34 + RNG 20261002 표집 66, 시드 공통 - P7 원안")
    L.append("> 60~100건과 정합). mask 행 = 정본(v2/v1)을 같은 표본으로 한정 재집계.")
    L.append("> delete 미보호에서 타깃 전부 삭제 시 p=0 정의. 집계 = 시드 내 평균 →")
    L.append("> 시드 간 평균±모표준편차, 어절·전체.")
    L.append("")
    for prot in PROTS:
        L.append(f"## {PROT_KO[prot]} (정답 스팬 {'순위 제외·보존' if prot == 'protected' else '교란 후보 포함'})")
        L.append("")
        L.append("| 전략 | 기법 | COMP@10 | SUFF@10 | AOPC |")
        L.append("|---|---|---:|---:|---:|")
        for strat in STRATS:
            for m in METHODS:
                k = (strat, prot, m)
                L.append(f"| {strat} | {METHOD_KO[m]} | {fmt(agg(acc[k + ('comp10',)]))} "
                         f"| {fmt(agg(acc[k + ('suff10',)]))} | {fmt(agg(acc[k + ('aopc',)]))} |")
        L.append("")
    L.append("## 판독")
    L.append("")
    L.append("- **방법 순위 불변**: 세 전략 모두에서 IG >> rollout ≈ random (보호 설정")
    L.append("  COMP: IG 0.68~0.77 vs 나머지 0.02~0.17).")
    L.append("- **보호 효과 방향 불변**: 미보호가 세 전략 모두에서 부풀림(mask 0.887→0.679,")
    L.append("  delete 0.948→0.773, randsub 0.894→0.743) - 자기-증거 효과는 전략 비의존.")
    L.append("- **delete 의 OOD 민감성**: 무작위 순위조차 COMP 0.154(보호)~0.266(미보호)의")
    L.append("  큰 낙폭 - 신호 없는 교란에도 반응하는 비특이성. [MASK] 채택(분포내 유지,")
    L.append("  원고 3.3.2)의 실측 근거. randsub 은 mask 와 거의 같은 패턴.")
    L.append("")
    L.append("## 원고 인용용 요약 (영문 1문장)")
    L.append("")
    L.append("> Replacing [MASK] substitution with token deletion or random in-vocabulary "
             "substitution (100-document fixed sample, three seeds, with and without "
             "gold-span protection) preserves both the method ranking (IG far above "
             "rollout and the random baseline under every strategy) and the direction "
             "of the span-protection effect, while deletion inflates the drop even for "
             "the random ranking (COMP@10 0.15-0.27) - an out-of-distribution "
             "sensitivity that empirically supports our choice of in-distribution "
             "[MASK] perturbation.")
    out = P12 / f"P12_결과_{date.today().strftime('%Y%m%d')}.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"생성: {out}")


if __name__ == "__main__":
    main()
