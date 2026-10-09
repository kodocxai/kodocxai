# -*- coding: utf-8 -*-
"""P15 보고서 - 귀속 5종 병렬 표 (신규 gxi·occl + 기존 3종 정본 복사).

기존 IG·rollout·random 행은 재계산하지 않고 정본 seed{N}_v2.jsonl(어절·전체)에서
복사한다(발주 지시). 전 표본·3시드.

사용: python report_p15.py
"""
from __future__ import annotations

import json
import statistics as st
from collections import defaultdict
from datetime import date
from pathlib import Path

P15 = Path(__file__).resolve().parent
RES = Path(r"D:\AIHub데이터\_exp\results")
SEEDS = (1, 2, 3)
ROWS = ["ig", "gxi", "occl", "rollout", "random"]
ROW_KO = {"ig": "IG (정본 복사)", "gxi": "Gradient×Input (신규)",
          "occl": "가림 leave-one-out (신규)", "rollout": "rollout (정본 복사)",
          "random": "무작위(하한, 정본 복사)"}


def iter_records():
    for s in SEEDS:
        for line in (P15 / f"P15_seed{s}.jsonl").read_text(encoding="utf-8").splitlines():
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

    L = [f"# P15 귀속 기법 추가 결과 ({date.today().isoformat()} 자동 생성)", ""]
    L.append("> `report_p15.py` 가 만든다. 손으로 고치지 말 것. 전 표본·3시드·어절·전체,")
    L.append("> v2 프로토콜 동일(보호·[MASK]·기하평균). gxi = |∇E·E 차원합|(IG abs 관례 통일),")
    L.append("> occl = 어절 [MASK] 시 스팬 확률 하락분(한 순전파 전 필드 공용). 기존 3종은")
    L.append("> 정본 복사(재계산 금지 준수). 집계 = 시드 내 평균 → 시드 간 평균±모표준편차.")
    L.append("")
    L.append("## 표 P15-1. 타당성 (수정본 표 6 병렬)")
    L.append("")
    L.append("| 기법 | IoU@1 | IoU@5 | F1@5 | Hit@1 | n/시드 |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for m in ROWS:
        ns = sorted({len(v) for v in n_inst[m].values()})
        L.append(f"| {ROW_KO[m]} | {fmt(agg(plaus[(m,'iou1')]))} | {fmt(agg(plaus[(m,'iou5')]))} "
                 f"| {fmt(agg(plaus[(m,'f1_5')]))} | {fmt(agg(plaus[(m,'hit1')]))} | {ns} |")
    L.append("")
    L.append("## 표 P15-2. 충실도 (수정본 표 5 병렬)")
    L.append("")
    L.append("| 기법 | COMP@10 | SUFF@10 | AOPC |")
    L.append("|---|---:|---:|---:|")
    for m in ROWS:
        L.append(f"| {ROW_KO[m]} | {fmt(agg(faith[(m,'comp10')]), 4)} "
                 f"| {fmt(agg(faith[(m,'suff10')]), 4)} | {fmt(agg(faith[(m,'aopc')]), 4)} |")
    L.append("")
    L.append("## 판독")
    L.append("")
    L.append("- **기법 스펙트럼 확장 후에도 기존 구도 유지**: 그래디언트·교란 계열 3종")
    L.append("  (IG·GxI·가림)은 전부 rollout·무작위를 전 지표에서 압도 - 「rollout 약세는")
    L.append("  기법 성질」(P11)과 합쳐져 어텐션 누적 계열만 약하다는 그림이 완성된다.")
    L.append("- **예상과 다른 결과(발주 지침대로 그대로 보고)**: GxI의 Hit@1 0.684가")
    L.append("  IG 0.402를 크게 상회(IoU@1도 0.329 > 0.175). 반면 IoU@5·F1@5·충실도는")
    L.append("  IG가 우위 - **GxI = 최상위 1점 포인팅에 강하고, IG = 스팬 커버리지·충실도에")
    L.append("  강하다**. 축마다 최강 기법이 다르다는 것은 다축 분리 보고 원칙(3.3.5)의")
    L.append("  실증이기도 하다.")
    L.append("- 주의(각주 재료): 가림 귀속은 충실도 프로토콜과 같은 교란 원시연산([MASK]")
    L.append("  낙폭)을 공유하므로 가림의 충실도 점수에는 순환성 우려를 명시할 것. 그럼에도")
    L.append("  AOPC 1위는 IG(0.609)로, 가림(0.603)이 자동으로 1위가 되지는 않았다.")
    L.append("")
    L.append("## 원고 인용용 요약 (영문 1문장)")
    L.append("")
    L.append("> Adding two further attribution methods leaves the framework's "
             "discrimination intact - both gradient- and perturbation-based methods "
             "(Gradient×Input, leave-one-out occlusion) far exceed rollout and the "
             "random baseline on every axis - while surfacing complementary "
             "strengths: Gradient×Input gives the sharpest top-1 pointing (Hit@1 "
             "0.684 vs 0.402 for IG) whereas IG retains the best top-5 span "
             "coverage and faithfulness (AOPC 0.609), supporting the framework's "
             "axis-separated reporting principle; occlusion shares its perturbation "
             "primitive with the faithfulness protocol, so its faithfulness scores "
             "carry a circularity caveat.")
    L.append("")
    L.append("재현 명령: `python eval_newattr.py --model ..\\runs\\seed{N}\\last "
             "--out P15_seed{N}.jsonl` (N=1,2,3) → `python report_p15.py`")
    out = P15 / f"P15_결과_{date.today().strftime('%Y%m%d')}.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"생성: {out}")


if __name__ == "__main__":
    main()
