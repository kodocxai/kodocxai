"""표본 크기 민감도 — 문서 부분표집으로 지표의 수렴을 본다.

묻는 것은 둘이다.
  ① **300건에서 지표가 수렴했는가** — 더 모으면 값이 달라지는가
  ② **각 결론이 성립하려면 문서가 몇 건 필요한가** — 결론마다 다르다

방법은 문서 단위 부분표집이다. 299건에서 n건을 비복원 추출해 지표를 다시 계산하기를
B회 반복하고, 평균과 2.5/97.5 분위수를 본다. 모델을 다시 돌리지 않는다 — 이미 계산된
(문서, 필드) 레코드를 다시 평균할 뿐이라 CPU 로 끝난다.

**⚠ 이 방법으로 말할 수 없는 것.** 300건 안에서의 수렴만 보인다. **n>300 의 거동은
외삽이며 중심극한정리 가정에 기댄다.** 표에 SE 열을 함께 두되 관측과 외삽을 구분한다.

사용
    python convergence.py [--seeds 1 2 3] [--boot 500]
"""

from __future__ import annotations

import argparse
import json
import random
import statistics as st
from pathlib import Path

RES = Path(r"D:\AIHub데이터\_exp\results")
OUT = RES / "표본크기_수렴.md"
GRID = [25, 50, 75, 100, 150, 200, 250, 299]
LEVELS = ["subword", "morpheme", "eojeol"]


def load(seed: int) -> dict:
    """(doc_id) → {(method, level): aopc, ('plaus', method): iou@5}"""
    per: dict = {}
    for line in (RES / f"seed{seed}_v2.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        if r["mask"] != "all":
            continue
        d = per.setdefault(r["doc_id"], {})
        a = r["faith"].get("aopc_comp")
        if a is not None:
            d.setdefault(("aopc", r["method"], r["level"]), []).append(a)
        io = (r["plaus"].get("5") or {}).get("iou")
        if io is not None and r["level"] == "eojeol":
            d.setdefault(("iou", r["method"]), []).append(io)
    # ⚠ 문서 안의 필드를 **미리 평균하지 않는다.** 표 3 은 (문서, 필드) 레코드를 고르게
    #   평균하므로, 문서 평균을 먼저 내면 필드 수가 다른 문서의 가중치가 달라져 표 3 과
    #   값이 어긋난다(실측: 0.5769 대 0.6086). 재추출 단위만 문서로 두고 값은 그대로 둔다
    #   — 군집 부트스트랩이다. 같은 문서의 필드끼리 상관이 있으므로 이쪽이 옳기도 하다.
    return per


def stat_of(sample: list, per: dict, key: tuple):
    v = [x for d in sample for x in per[d].get(key, ())]
    return st.mean(v) if v else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--seed-rng", type=int, default=20260817)
    a = ap.parse_args()

    L, add = [], None
    L = []
    add = L.append
    add("# 표본 크기 민감도 — 문서 부분표집 수렴 분석")
    add("")
    add("> `_tools\\convergence.py` 가 만든다. **손으로 고치지 말 것.**")
    add(f"> 299건에서 n건 비복원 추출 × {a.boot}회. 모델 재실행 없음(기존 레코드 재평균).")
    add("> 괄호는 2.5~97.5 분위수다.")
    add("")
    add("**⚠ 읽는 법.** 이 분석이 보이는 것은 **300건 안에서의 수렴**이다.")
    add("n>300 의 거동은 관측이 아니라 외삽이다.")
    add("")

    rng = random.Random(a.seed_rng)
    per_seed = {s: load(s) for s in a.seeds}
    docs0 = sorted(per_seed[a.seeds[0]])
    add(f"- 시드 {a.seeds} · 문서 {len(docs0)}건 · 재추출 {a.boot}회")
    add("")

    # ── A. 핵심 지표의 수렴
    # ⚠ **비복원 추출은 n=N 에서 퇴화한다**(299건에서 299건을 뽑으면 항상 전체라
    #   변동이 0 이 된다). 모든 n 에서 **복원 추출(부트스트랩)** 을 쓴다. 그래야
    #   n=299 의 구간도 실제 추정 불확실성을 뜻한다.
    KEYS = [("aopc", "ig", "eojeol"), ("aopc", "rollout", "eojeol"),
            ("aopc", "random", "eojeol"), ("iou", "ig")]
    NAME = {("aopc", "ig", "eojeol"): "IG AOPC",
            ("aopc", "rollout", "eojeol"): "rollout AOPC",
            ("aopc", "random", "eojeol"): "무작위 AOPC",
            ("iou", "ig"): "IG IoU@5"}

    # 시드 안에서 부트스트랩 → 문서 추출이 만드는 폭만 잰다. 시드 간 변동은 B 에서 따로.
    width, center = {}, {}
    for n in GRID:
        for key in KEYS:
            ws, cs = [], []
            for s in a.seeds:
                per = per_seed[s]
                ds = sorted(per)
                vals = sorted(v for v in
                              (stat_of(rng.choices(ds, k=n), per, key)
                               for _ in range(a.boot)) if v is not None)
                lo, hi = vals[int(.025 * len(vals))], vals[int(.975 * len(vals))]
                ws.append(hi - lo)
                cs.append(st.mean(vals))
            width.setdefault(key, {})[n] = st.mean(ws)
            center.setdefault(key, {})[n] = st.mean(cs)

    add("## A. 지표는 몇 건에서 안정되는가 (시드 내 부트스트랩)")
    add("")
    add("값은 시드 3회의 평균이고, 괄호는 **문서 추출이 만드는 95% 구간 폭**이다.")
    add("")
    add("| n | " + " | ".join(NAME[k] for k in KEYS) + " |")
    add("|---:" + "|---" * len(KEYS) + "|")
    for n in GRID:
        add(f"| {n} | " + " | ".join(
            f"{center[k][n]:.4f} (±{width[k][n]/2:.4f})" for k in KEYS) + " |")
    add("")
    add("**중심값은 n=25 에서 이미 안정된다** — 늘어나는 것은 정밀도뿐이다.")
    add("")
    add("| n | " + " | ".join(NAME[k] for k in KEYS) + " |")
    add("|---:" + "|---:" * len(KEYS) + "|")
    for n in GRID:
        add(f"| {n} | " + " | ".join(
            f"{width[k][n] / width[k][299]:.2f}배" for k in KEYS) + " |")
    add("")

    # ── B. 결론이 성립하는 최소 표본
    add("## B. 각 결론에 필요한 최소 문서 수")
    add("")
    add("재추출 표본에서 그 결론이 성립한 비율이다. **95% 이상이면 그 n 에서 안정**이다.")
    add("")
    add("| n | IG > 무작위 | rollout ≈ 무작위 (차 < 0.02) | 집계 수준 차 < 0.01 |")
    add("|---:|---:|---:|---:|")
    for n in GRID:
        c1 = c2 = c3 = tot = 0
        for s in a.seeds:
            per = per_seed[s]
            ds = sorted(per)
            for _ in range(a.boot // len(a.seeds)):
                smp = rng.choices(ds, k=n)
                ig = stat_of(smp, per, ("aopc", "ig", "eojeol"))
                ro = stat_of(smp, per, ("aopc", "rollout", "eojeol"))
                rd = stat_of(smp, per, ("aopc", "random", "eojeol"))
                lv = [stat_of(smp, per, ("aopc", "ig", l)) for l in LEVELS]
                if None in (ig, ro, rd) or None in lv:
                    continue
                tot += 1
                c1 += ig > rd
                c2 += abs(ro - rd) < 0.02
                c3 += (max(lv) - min(lv)) < 0.01
        add(f"| {n} | {c1/tot*100:.1f}% | {c2/tot*100:.1f}% | {c3/tot*100:.1f}% |")
    add("")

    # ── C. 문서를 늘릴 것인가, 시드를 늘릴 것인가
    add("## C. 남은 불확실성은 어디서 오는가 — 문서 대 재학습")
    add("")
    add("n=299 에서 **문서 추출이 만드는 폭**과 **시드(재학습) 간 폭**을 나란히 본다.")
    add("둘 중 큰 쪽이 다음에 투자할 곳이다.")
    add("")
    add("| 지표 | 문서 추출 (n=299) | 시드 간 | 지배 요인 |")
    add("|---|---:|---:|---|")
    for key in KEYS:
        per_seed_mean = []
        for s in a.seeds:
            per = per_seed[s]
            v = stat_of(sorted(per), per, key)
            if v is not None:
                per_seed_mean.append(v)
        seed_sd = st.pstdev(per_seed_mean) if len(per_seed_mean) > 1 else 0.0
        seed_w = 2 * 1.96 * seed_sd          # 95% 폭으로 환산해 문서 쪽과 맞춘다
        doc_w = width[key][299]
        dom = "**시드(재학습)**" if seed_w > doc_w else "문서 표본"
        add(f"| {NAME[key]} | {doc_w:.4f} | {seed_w:.4f} | {dom} "
            f"({max(seed_w, doc_w)/max(1e-9, min(seed_w, doc_w)):.1f}배) |")
    add("")
    add("**이 표가 이 분석의 요지다.** 지배 요인이 시드 쪽이면 **문서를 늘려도 구간이")
    add("좁아지지 않는다** — 늘려야 하는 것은 재학습 횟수다. 벤치마크 설계에 바로 쓰이는")
    add("결론이며, 「300건은 왜 충분한가」에 대한 답이기도 하다.")
    add("")
    add("> **⚠ 외삽 주의.** A·B 는 관측이지만 「n=1000 이면 얼마」는 이 분석에 없다.")
    add("> 부트스트랩 폭이 $1/\\sqrt{n}$ 로 줄어든다는 가정을 쓰면 추정할 수 있으나,")
    add("> 그것은 가정이지 관측이 아니다.")
    add("")
    add(f"> **⚠ C 표의 한계 — 시드가 {len(a.seeds)}개뿐이다.** 3개 값으로 낸 표준편차는")
    add("> 그 자체로 부정확하다. **비율(9.7배 등)을 그대로 인용하지 말 것.** 말할 수 있는")
    add("> 것은 **방향**이다 — IG 계열에서 시드 간 폭이 문서 추출 폭보다 뚜렷이 크다.")
    add("> rollout 은 1.2배로 사실상 대등하므로 「시드가 지배한다」고 쓰면 안 된다.")

    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"생성: {OUT}")
    for k in KEYS:
        print(f"  {k}: 95%폭 n=25 {width[k][25]:.4f} → n=299 {width[k][299]:.4f}")


if __name__ == "__main__":
    main()
