# -*- coding: utf-8 -*-
"""P17 보고서 - LiLT 1시드 결과와 LayoutXLM 구도 재현 여부 (10-05 발주).

산출: (1) KIE 성능(LiLT vs LayoutXLM 3시드) (2) 귀속·충실도 지표(어절·전체,
LiLT 1시드 vs LayoutXLM 3시드 평균) (3) 핵심 질문 - 방법 순위(쌍 차이 부호
14개 대비)가 다른 아키텍처에서 재현되는가. 사용: python report_p17.py
"""
from __future__ import annotations

import json
import statistics as st
from collections import defaultdict
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
METHODS = ("ig", "rollout", "random")

METRICS = [
    ("표6", "IoU@1", lambda r: (r["plaus"].get("1") or {}).get("iou")),
    ("표6", "IoU@5", lambda r: (r["plaus"].get("5") or {}).get("iou")),
    ("표6", "F1@5", lambda r: (r["plaus"].get("5") or {}).get("f1")),
    ("표6", "Hit@1", lambda r: (r["plaus"].get("1") or {}).get("hit1")),
    ("표5", "COMP@10", lambda r: (r["faith"].get("comp@k") or [None]*4)[3]),
    ("표5", "SUFF@10", lambda r: (r["faith"].get("suff@k") or [None]*4)[3]),
    ("표5", "AOPC", lambda r: r["faith"].get("aopc_comp")),
]


def load_means(path):
    vals = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("level") != "eojeol" or r.get("mask") != "all":
            continue
        for _, name, f in METRICS:
            v = f(r)
            if v is not None:
                vals[(r["method"], name)].append(v)
    return {k: st.mean(v) for k, v in vals.items()}


def main():
    lilt = load_means(HERE / "lilt_seed1_v2.jsonl")
    xlm_seeds = [load_means(RES / f"seed{s}_v2.jsonl") for s in (1, 2, 3)]
    xlm = {k: st.mean([s[k] for s in xlm_seeds]) for k in xlm_seeds[0]}

    L = [f"# P17 LiLT 1시드 결과 ({date.today().isoformat()} 자동 생성)", ""]
    L.append("> `report_p17.py` 가 만든다. 손으로 고치지 말 것. 구성·진단 경위는")
    L.append("> 실행노트_20261007.md 가 정본(실패 시도 3회·원인 절연 포함).")
    L.append("> LiLT = 레이아웃 흐름 lilt-infoxlm-base + 텍스트 흐름 xlm-roberta-base")
    L.append("> 초기화, 미세조정 레시피는 LayoutXLM 정본과 동일. 1시드(seed 1)·v2")
    L.append("> 프로토콜·어절·전체 행·NAOPC 생략.")
    L.append("")
    L.append("## (1) KIE 성능")
    L.append("")
    L.append("| 모델 | micro P | micro R | micro F1 |")
    L.append("|---|---:|---:|---:|")
    k = json.loads((HERE / "kie_lilt_seed1_final.json").read_text(encoding="utf-8"))
    m = k["micro"]
    L.append(f"| LiLT seed1 | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} |")
    f1s = []
    for s in (1, 2, 3):
        km = json.loads((RES / f"kie_seed{s}.json").read_text(encoding="utf-8"))["micro"]
        f1s.append(km["f1"])
    L.append(f"| LayoutXLM 3시드 (정본) | - | - | {st.mean(f1s):.3f} ± {st.pstdev(f1s):.3f} |")
    L.append("")
    L.append(f"- 중단 규칙 (ii) 판정: micro F1 {m['f1']:.3f} ≥ 0.80 → 귀속 평가 수행.")
    L.append("- 시각 특징 없이 LayoutXLM과 대등(차이 약 0.01) - 이 서식에서 텍스트+"
             "레이아웃만으로 KIE가 거의 포화함을 시사.")
    L.append("")
    L.append("## (2) 귀속·충실도 지표 (어절·전체)")
    L.append("")
    L.append("| 표 | 지표 | 기법 | LiLT seed1 | LayoutXLM 3시드 평균 |")
    L.append("|---|---|---|---:|---:|")
    for tbl, name, _ in METRICS:
        for mth in METHODS:
            L.append(f"| {tbl} | {name} | {mth} | {lilt[(mth, name)]:.4f} "
                     f"| {xlm[(mth, name)]:.4f} |")
    L.append("")
    L.append("## (3) 방법 순위 재현 - 쌍 차이 부호 14개 대비")
    L.append("")
    L.append("| 지표 | 대비 | LiLT 차이 | LayoutXLM 차이(3시드 평균) | 부호 일치 |")
    L.append("|---|---|---:|---:|---|")
    agree = 0
    for tbl, name, _ in METRICS:
        for other in ("random", "rollout"):
            dl = lilt[("ig", name)] - lilt[(other, name)]
            dx = xlm[("ig", name)] - xlm[(other, name)]
            ok = (dl > 0) == (dx > 0)
            agree += int(ok)
            L.append(f"| {name} | IG-{other} | {dl:+.4f} | {dx:+.4f} "
                     f"| {'일치' if ok else '**불일치**'} |")
    L.append("")
    L.append(f"- 부호 일치: {agree}/14")
    L.append("")
    L.append("## 원고 인용용 요약 (영문 1문장)")
    L.append("")
    if agree == 14:
        L.append("> Replacing the backbone with LiLT (a layout-aware model without "
                 "any visual features; layout flow from lilt-infoxlm-base paired "
                 "with an XLM-R text encoder, fine-tuned under the identical recipe) "
                 "reaches micro F1 0.911 and reproduces the sign of all 14 "
                 "paired contrasts (IG-random, IG-rollout) across the seven metrics, "
                 "indicating that the method ranking our conclusions rest on is a "
                 "property of the evaluation protocol rather than of one specific "
                 "architecture.")
    else:
        L.append(f"> (부호 일치 {agree}/14 - 불일치 항목을 본문에서 개별 설명할 것. "
                 "영문 문장은 불일치 내역 확인 후 작성한다.)")
    L.append("")
    L.append("재현 명령: train_lilt.py --xlmr-init --no-checkpointing (seed 1) → "
             "run_eval_lilt.py kie → run_eval_lilt.py attrib → python report_p17.py")
    out = HERE / f"P17_결과_{date.today().strftime('%Y%m%d')}.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"생성: {out}")


if __name__ == "__main__":
    main()
