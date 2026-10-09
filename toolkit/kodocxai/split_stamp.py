"""소분석 — 직인 겹침 문서 대 정상 문서로 표 5 를 갈라 본다 (원고 §5.2 자연 실험).

집필세션 요청(08-16 13:49, 저자 승인). 묻는 것은 하나다.
**발신명의가 직인에 가려 판독 불가(status = string_na)인 문서에서, 모델은 다른 모달로
갈아타는가?** 표 5 는 전 문서 평균이라 이 대비가 평균 안에 묻혀 있다.

「자연 실험」이라 부를 수 있는 이유는 직인 겹침이 **연구자가 만든 조작이 아니라 원자료가
가진 변이**이기 때문이다. 다만 무작위 배정이 아니므로(직인은 특정 연도·기관에 몰릴 수
있다) 인과 주장은 못 한다 — 문서 수와 함께 그 한계를 같이 적어야 한다.

**새 계산이 없다.** `seed{N}_v2.jsonl` 의 문서별 레코드를 부분집합으로 갈라 평균할 뿐이다.
표 5 와 **같은 정의**를 쓴다 — IG · 어절 · mask=all, 텍스트는 comp@k 의 마지막 값(COMP@10).

사용
    python split_stamp.py [--seeds 1 2 3]
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

PILOT = Path(r"D:\AIHub데이터\_pilot")
RES = Path(r"D:\AIHub데이터\_exp\results")
FIELDS = ["제목", "수신", "문서번호", "시행일자", "발신명의"]
OUT = RES / "소분석_직인겹침.md"


def stamped_docs() -> set:
    """발신명의가 직인에 가려 판독 불가인 문서. append-only 라 나중 줄이 이긴다."""
    gold = {}
    for line in (PILOT / "annotations_park.jsonl").open(encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            gold[r["doc_id"]] = r
    return {d for d, r in gold.items()
            if ((r.get("fields") or {}).get("발신명의") or {}).get("status") == "string_na"}, len(gold)


def load(seeds: list) -> dict:
    out = {}
    for s in seeds:
        p = RES / f"seed{s}_v2.jsonl"
        if not p.exists():
            continue
        rows = []
        for line in p.open(encoding="utf-8"):
            r = json.loads(line)
            if r["level"] == "eojeol" and r["method"] == "ig" and r["mask"] == "all":
                rows.append(r)
        out[s] = rows
    return out


def cell(rows_by_seed: dict, field: str, docs: set, get) -> tuple:
    """시드 안에서 문서 평균 → 시드 간 평균±표준편차. 표 5 와 같은 순서다."""
    per = []
    ns = []
    for s, rows in rows_by_seed.items():
        v = [get(r) for r in rows if r["field"] == field and r["doc_id"] in docs]
        v = [x for x in v if x is not None]
        if v:
            per.append(st.mean(v))
            ns.append(len(v))
    if not per:
        return None, None, 0
    sd = st.pstdev(per) if len(per) > 1 else 0.0
    return st.mean(per), sd, (round(st.mean(ns)) if ns else 0)


def fmt(m, sd) -> str:
    return "—" if m is None else f"{m:.4f} ± {sd:.4f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    a = ap.parse_args()

    na, n_ann = stamped_docs()
    rows = load(a.seeds)
    if not rows:
        raise SystemExit("seed*_v2.jsonl 을 찾지 못했다")

    seen = {r["doc_id"] for v in rows.values() for r in v}
    stamped = na & seen
    normal = seen - na

    METRICS = [
        ("텍스트 (COMP@10)", lambda r: (r["faith"].get("comp@k") or [None] * 4)[-1]),
        ("지터 δ=25", lambda r: r["faith"].get("layout_jitter@25")),
        ("지터 δ=67", lambda r: r["faith"].get("layout_jitter@67")),
        ("지터 δ=134", lambda r: r["faith"].get("layout_jitter@134")),
        ("레이아웃 (교환)", lambda r: r["faith"].get("layout_swap")),
        ("이미지 (occlusion)", lambda r: r["faith"].get("image_occlusion")),
        ("p_full", lambda r: r.get("p_full")),
    ]

    L = []
    add = L.append
    add("# 소분석 — 직인 겹침 문서 대 정상 문서 (원고 §5.2)")
    add("")
    add("> `_tools\\split_stamp.py` 가 만든다. **손으로 고치지 말 것.**")
    add("> 집필세션 요청(08-16 13:49, 저자 승인). 표 5 와 **같은 정의**를 쓴다 —")
    add("> IG · 어절 · mask=all, 텍스트는 COMP@10. 시드 안에서 문서 평균 → 시드 간 평균±표준편차.")
    add("")
    add(f"- 주석 문서 {n_ann}건 중 **발신명의 `string_na`(직인 겹침) {len(na)}건**")
    add(f"- 평가에 등장한 문서 {len(seen)}건 → **직인 {len(stamped)}건 · 정상 {len(normal)}건**")
    add(f"- 시드 {sorted(rows)}")
    add("")

    for f in FIELDS:
        add(f"## {f}")
        add("")
        add("| 교란 | 직인 겹침 | 정상 | 차이(직인−정상) |")
        add("|---|---:|---:|---:|")
        for name, get in METRICS:
            ms, ss, ns = cell(rows, f, stamped, get)
            mn, sn, nn = cell(rows, f, normal, get)
            d = "—" if (ms is None or mn is None) else f"{ms - mn:+.4f}"
            add(f"| {name} | {fmt(ms, ss)} | {fmt(mn, sn)} | {d} |")
        add("")
        ns = cell(rows, f, stamped, lambda r: r.get("p_full"))[2]
        nn = cell(rows, f, normal, lambda r: r.get("p_full"))[2]
        add(f"시드당 문서 수 — 직인 {ns}건 · 정상 {nn}건")
        add("")

    add("---")
    add("")
    add("## 읽을 때 주의")
    add("")
    add("**무작위 배정이 아니다.** 직인 겹침은 원자료가 가진 변이이지 연구자가 만든 조작이")
    add("아니므로, 특정 연도·기관에 몰려 있을 수 있다. **인과가 아니라 연관으로 쓸 것.**")
    add("직인 문서 수가 적어(위 표) 시드 간 표준편차가 정상 문서보다 크게 나오는 것이")
    add("정상이다 — 차이를 읽을 때 표준편차와 겹치는지 먼저 볼 것.")
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"생성: {OUT}")
    print(f"직인 겹침 {len(stamped)}건 · 정상 {len(normal)}건")
    for f in FIELDS:
        ms = cell(rows, f, stamped, METRICS[0][1])
        mn = cell(rows, f, normal, METRICS[0][1])
        sw_s = cell(rows, f, stamped, METRICS[4][1])
        sw_n = cell(rows, f, normal, METRICS[4][1])
        print(f"  {f:5s} 텍스트 {fmt(ms[0], ms[1])} vs {fmt(mn[0], mn[1])}   "
              f"교환 {fmt(sw_s[0], sw_s[1])} vs {fmt(sw_n[0], sw_n[1])}")


if __name__ == "__main__":
    main()
