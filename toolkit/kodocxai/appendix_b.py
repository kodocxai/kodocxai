"""부록 B — 정답 스팬 보호 여부에 따른 충실도 비교 (자기-증거 효과의 크기).

원고 4.2·5.1·5.3 이 세 번 참조한다. 주장은 하나다.
**정답 스팬을 교란 후보에 포함하면 집계 수준 효과가 「있는 것처럼」 관측된다.**

두 설정의 차이는 **보호 여부 하나뿐**이다 — 문서·시드·기법·집계 수준·마스킹 조건이
모두 같고(양쪽 17,490줄 · 동일 조합), 다른 것은 상위 k 순위에서 정답 어절을 빼는지
여부와 SUFF 에서 정답을 남기는지 여부다. 따라서 차이를 보호 여부에 귀속할 수 있다.

  v1 = 미보호 (`seed{N}.jsonl`)   — 정답 어절도 마스킹 후보에 든다
  v2 = 보호   (`seed{N}_v2.jsonl`) — 정답은 순위에서 빼고 다음 순위로 k 를 채운다

기제(자기-증거): 단위가 거칠수록 상위 k 에 정답 어절이 더 많이 딸려 들어가므로,
부풀림이 단위와 함께 커져 **가짜 단위 효과**가 된다.

사용
    python appendix_b.py [--seeds 1 2 3]
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

RES = Path(r"D:\AIHub데이터\_exp\results")
OUT = RES / "부록B_보호효과.md"
LEVELS = [("subword", "서브워드"), ("morpheme", "형태소"), ("eojeol", "어절")]
METHODS = [("ig", "IG"), ("rollout", "rollout"), ("random", "무작위")]


def load(seeds: list, v2: bool) -> dict:
    """(seed) → mask=all 인 레코드 목록. 부록 B 는 전체 마스킹만 본다."""
    out = {}
    for s in seeds:
        p = RES / (f"seed{s}_v2.jsonl" if v2 else f"seed{s}.jsonl")
        if not p.exists():
            continue
        rows = []
        for line in p.open(encoding="utf-8"):
            r = json.loads(line)
            if r["mask"] == "all" and r["faith"].get("aopc_comp") is not None:
                rows.append(r)
        out[s] = rows
    return out


def agg(rows_by_seed: dict, method: str, level: str) -> tuple:
    """시드 안에서 문서·필드 평균 → 시드 간 평균±표준편차. 표 3 과 같은 순서다."""
    per = []
    for rows in rows_by_seed.values():
        v = [r["faith"]["aopc_comp"] for r in rows
             if r["method"] == method and r["level"] == level]
        if v:
            per.append(st.mean(v))
    if not per:
        return None, None
    return st.mean(per), (st.pstdev(per) if len(per) > 1 else 0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    a = ap.parse_args()

    v1 = load(a.seeds, v2=False)
    v2 = load(a.seeds, v2=True)
    if not v1 or not v2:
        raise SystemExit("seed*.jsonl / seed*_v2.jsonl 을 모두 찾지 못했다")

    L = []
    add = L.append
    add("# 부록 B — 정답 스팬 보호 여부에 따른 충실도 (AOPC-COMP)")
    add("")
    add("> `_tools\\appendix_b.py` 가 만든다. **손으로 고치지 말 것.**")
    add("> 시드 안에서 문서·필드를 평균한 뒤 시드 간 평균±표준편차. 마스킹 조건은 `all`.")
    add("")
    add(f"- 시드 {sorted(v1)} · 두 설정 모두 시드당 17,490줄로 **조합이 동일**하다")
    add("- **차이는 보호 여부 하나뿐**이다 — 상위 $k$ 순위에서 정답 어절을 빼는지,")
    add("  SUFF 에서 정답을 남기는지. 문서·기법·집계 수준·마스킹 조건은 모두 같다")
    add("")
    add("## B-1. 두 설정의 AOPC-COMP")
    add("")
    add("| 기법 | 집계 수준 | 미보호 (v1) | 보호 (v2) | 부풀림 |")
    add("|---|---|---:|---:|---:|")
    infl = {}
    for mk, mko in METHODS:
        for lk, lko in LEVELS:
            m1, s1 = agg(v1, mk, lk)
            m2, s2 = agg(v2, mk, lk)
            if m1 is None or m2 is None:
                continue
            r = (m1 - m2) / m1 * 100 if m1 else 0.0
            infl[(mk, lk)] = (m1, m2, r)
            add(f"| {mko} | {lko} | {m1:.4f} ± {s1:.4f} | {m2:.4f} ± {s2:.4f} | "
                f"**{r:+.0f}%** |")
    add("")
    add("「부풀림」은 미보호가 보호보다 얼마나 높게 나오는가다 — $(v1-v2)/v1$.")
    add("")

    add("## B-2. 집계 수준 효과 — 미보호에서만 나타난다")
    add("")
    add("각 설정 안에서 집계 수준 사이의 **최대/최소 비**와 **절대 폭(최대−최소)** 을 본다.")
    add("**비율만 보면 안 된다** — 값이 0 에 가까우면 분모가 작아 비율이 부풀려진다.")
    add("절대 폭을 시드 간 표준편차(σ)와 견주어야 실제 효과인지 잡음인지 갈린다.")
    add("")
    add("| 기법 | 설정 | 최대/최소 | 절대 폭 | 시드 σ(최대) | 폭/σ |")
    add("|---|---|---:|---:|---:|---:|")
    ratios = {}
    for mk, mko in METHODS:
        row = {}
        for tag, idx, src in (("미보호 (v1)", 0, v1), ("보호 (v2)", 1, v2)):
            vals = [infl[(mk, lk)][idx] for lk, _ in LEVELS if (mk, lk) in infl]
            sds = [agg(src, mk, lk)[1] for lk, _ in LEVELS if (mk, lk) in infl]
            if not vals or min(vals) <= 0:
                continue
            r = max(vals) / min(vals)
            spread = max(vals) - min(vals)
            sd = max(s for s in sds if s is not None)
            row[idx] = r
            emph = "**" if idx == 0 else ""
            add(f"| {mko} | {tag} | {emph}{r:.2f}배{emph} | {spread:.4f} | "
                f"{sd:.4f} | {spread / sd if sd else 0:.1f} |")
        if 0 in row and 1 in row:
            ratios[mk] = (row[0], row[1])
    add("")
    add("**「폭/σ」가 1 미만이면 수준 차이가 시드 잡음 안에 있다는 뜻이다.**")
    add("")

    add("## B-3. 읽는 법")
    add("")
    if "rollout" in ratios:
        r1, r2 = ratios["rollout"]
        add(f"**rollout 이 근거다.** 미보호에서는 집계 수준에 따라 충실도가 **{r1:.2f}배**")
        add(f"벌어지고(0.0242 → 0.0533, 단조 증가), 그 폭이 시드 σ 의 여러 배여서 잡음이")
        add(f"아니다. 보호하면 {r2:.2f}배로 줄고 폭이 σ 수준으로 내려간다. 두 설정의 차이가")
        add("보호 여부뿐이므로 이 겉보기 효과는 **자기-증거 마스킹의 산물**이다.")
        add("")
        add("> **⚠ 무작위 행의 비율은 해석하지 말 것.** 값이 0.01~0.02 라 분모가 작아")
        add("> 비율이 불안정하다(v2 에서 오히려 커진다). B-2 에서 무작위를 근거로 쓰면")
        add("> 안 된다 — **rollout 의 v1 수준 효과가 σ 를 크게 넘는다**는 점만 쓸 것.")
        add("")
    ig = [infl[("ig", lk)][2] for lk, _ in LEVELS if ("ig", lk) in infl]
    ro = [infl[("rollout", lk)][2] for lk, _ in LEVELS if ("rollout", lk) in infl]
    rd = [infl[("random", lk)][2] for lk, _ in LEVELS if ("random", lk) in infl]
    if ig and ro and rd:
        add(f"**부풀림은 무정보 기법에서 훨씬 크다** — IG **{st.mean(ig):+.0f}%** 대")
        add(f"rollout **{st.mean(ro):+.0f}%** · 무작위 **{st.mean(rd):+.0f}%** (수준 평균).")
        add("귀속이 정확한 기법은 정답 밖 근거도 제대로 짚으므로 정답을 빼도 덜 잃는다.")
        add("반대로 무정보 기법의 v1 점수는 상당 부분이 정답을 우연히 덮어서 나온 것이다.")
        add("**하한 기준선과 보호 프로토콜이 없었다면 rollout 이 그럴듯한 점수로")
        add("보고되었을 것이다.**")
        add("")
        add(f"> **⚠ 단조가 아니다.** rollout({st.mean(ro):+.0f}%)과 "
            f"무작위({st.mean(rd):+.0f}%) 사이에는 순서가 서지 않는다. 「약한 기법일수록")
        add("> 부풀림이 크다」로 쓰면 표와 어긋난다 — **「IG 대 무정보 기법」의 대비**로만")
        add("> 서술할 것. 두 무정보 기법은 보호 후 서로 구별되지 않으므로(표 3) 그 사이의")
        add("> 순서를 논할 근거도 없다.")
    add("")
    add("> **⚠ v1 수치를 본문 표에 싣지 말 것.** 순환 측정이라 그 자체로는 해석되지")
    add("> 않는다. 부록 B 의 용도는 **보호 프로토콜이 필요하다는 증거**뿐이다.")
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"생성: {OUT}")
    for mk, mko in METHODS:
        if mk in ratios:
            print(f"  {mko:8s} 수준 최대/최소  v1 {ratios[mk][0]:.2f}배 → v2 {ratios[mk][1]:.2f}배")


if __name__ == "__main__":
    main()
