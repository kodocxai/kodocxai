# -*- coding: utf-8 -*-
"""표 7 레이아웃 열 재집계 (2026-10-26, R1 ②). 기존 report_exp.py 의 agg 와 같은 방식:
시드 안에서 인스턴스 평균 → 시드 간 평균 ± 모표준편차(pstdev).

출력: results/layout_v3_report.md (구 값 ↔ 새 값 대조, 제외 건수, 검증 일치율)
사용: python report_layout_v3.py
"""
import json, statistics as st, sys
from collections import defaultdict
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')

EXP = Path(__file__).resolve().parent
FIELDS = ["제목", "수신", "문서번호", "시행일자", "발신명의"]
EN = {"제목": "Title", "수신": "Recipient", "문서번호": "Document number",
      "시행일자": "Issuance date", "발신명의": "Issuing authority"}
SEEDS = [1, 2, 3]
JK = ["layout_jitter@25", "layout_jitter@67", "layout_jitter@134", "layout_swap"]


def load(seed):
    p = EXP / "results" / f"layout_v3_seed{seed}.jsonl"
    rows = {}
    for line in p.open(encoding="utf-8"):
        r = json.loads(line); rows[(r["doc_id"], r["field"])] = r
    return rows


def agg(per_seed_vals):
    """{seed: [values]} -> (mean of seed means, pstdev, n_total)"""
    means = [st.mean(v) for v in per_seed_vals.values() if v]
    n = sum(len(v) for v in per_seed_vals.values())
    return (st.mean(means), st.pstdev(means) if len(means) > 1 else 0.0, n) if means else None


def fmt(t, d=4):
    return "—" if t is None else f"{t[0]:.{d}f} ± {t[1]:.{d}f}"


def main():
    data = {s: load(s) for s in SEEDS}
    out = []
    add = out.append
    add("# 표 7 레이아웃 열 재집계 (R1 ②, 2026-10-26)\n")
    add("시드 안에서 인스턴스 평균 → 시드 3회 평균 ± 모표준편차. 새 규칙 = 지터 어절 단위, 교환 상대 = 다른 필드 주석 스팬 토큰(읽기 순서 앞부터, 짧은 쪽 길이까지).\n")

    # 검증
    n_all = n_pf = n_old = 0
    for s in SEEDS:
        for r in data[s].values():
            c = r["check"]
            if "p_full_ref" in c:
                n_all += 1
                if abs(r["p_full"] - c["p_full_ref"]) < 1e-9: n_pf += 1
                if c.get("layout_swap_oldrule") is not None and c.get("layout_swap_ref") is not None \
                        and abs(c["layout_swap_oldrule"] - c["layout_swap_ref"]) < 1e-9: n_old += 1
    add(f"## 검증\n\n- 인스턴스 {n_all}건(3시드 합). p_full 기존 결과와 일치 {n_pf}건, 구 규칙 교환값 일치 {n_old}건.\n")
    cnt = {s: len(data[s]) for s in SEEDS}
    add(f"- 시드별 인스턴스 수: {cnt}\n")

    # 제외 건수
    add("## 교환 조건 제외(상대 없음) 건수\n")
    add("| 필드 | " + " | ".join(f"seed{s}" for s in SEEDS) + " |")
    add("|---|" + "---:|" * len(SEEDS))
    tot_ex = {s: 0 for s in SEEDS}
    for f in FIELDS:
        row = []
        for s in SEEDS:
            ex = sum(1 for r in data[s].values() if r["field"] == f and r["faith"].get("layout_swap") is None)
            tot_ex[s] += ex; row.append(str(ex))
        add(f"| {f} | " + " | ".join(row) + " |")
    add(f"| 합계 | " + " | ".join(str(tot_ex[s]) for s in SEEDS) + " |\n")
    # 부분 교환(상대 수 < 정답 토큰 수)
    part = sum(1 for s in SEEDS for r in data[s].values() if r["faith"].get("layout_swap") is not None and r["swap_partner_n"] < r["n_gold_tk"])
    add(f"- 상대 토큰 수가 정답 토큰 수보다 적어 일부만 교환된 인스턴스: {part}건(3시드 합).\n")

    # 표
    add("## 표 7 레이아웃 열: 구 값(기존 결과 재집계) → 새 값\n")
    add("| Field | Jitter δ=25 구 | 새 | Jitter δ=67 구 | 새 | Jitter δ=134 구 | 새 | Coordinate swap 구 | 새 |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    newtab = {}
    for f in FIELDS + ["ALL"]:
        cells = []
        for k in JK:
            old = {s: [r["check"].get(k + "_ref") for r in data[s].values()
                       if (f == "ALL" or r["field"] == f) and r["check"].get(k + "_ref") is not None] for s in SEEDS}
            new = {s: [r["faith"].get(k) for r in data[s].values()
                       if (f == "ALL" or r["field"] == f) and r["faith"].get(k) is not None] for s in SEEDS}
            o, n = agg(old), agg(new)
            newtab[(f, k)] = n
            cells += [fmt(o), fmt(n)]
        add(f"| {EN.get(f, '전체')} | " + " | ".join(cells) + " |")
    add("")
    # 발신명의 교환: 정상 vs 직인 겹침은 split_stamp.py 가 별도 처리하므로 여기서는 생략.
    add("## 새 값 (논문 표 7 삽입용, 4자리)\n")
    add("| Field | Jitter δ = 25 | Jitter δ = 67 | Jitter δ = 134 | Coordinate swap |")
    add("|---|---:|---:|---:|---:|")
    for f in FIELDS:
        add(f"| {EN[f]} | " + " | ".join(fmt(newtab[(f, k)]) for k in JK) + " |")
    add("")
    # 새 값 (3자리) 본문 인용용 + json
    (EXP / "results" / "layout_v3_table.json").write_text(json.dumps(
        {f"{f}|{k}": v for (f, k), v in newtab.items()}, ensure_ascii=False, indent=1), encoding="utf-8")
    (EXP / "results" / "layout_v3_report.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
