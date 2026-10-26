# -*- coding: utf-8 -*-
"""직인 겹침(string_na) 대 정상 문서의 발신명의 좌표 교환 반응, 새 규칙(layout_v3) 기준.
_tools/split_stamp.py 와 같은 정의(시드 안 문서 평균 → 시드 간 평균±모표준편차). 7.2절의
"0.268 vs 0.008" 을 대체할 값을 낸다. 사용: python split_stamp_v3.py
"""
import json, statistics as st, sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
PILOT = Path(r"D:\AIHub데이터\_pilot"); RES = Path(r"D:\AIHub데이터\_exp\results")
FIELDS = ["제목", "수신", "문서번호", "시행일자", "발신명의"]

gold = {}
for line in (PILOT / "annotations_park.jsonl").open(encoding="utf-8"):
    if line.strip():
        r = json.loads(line); gold[r["doc_id"]] = r
stamped = {d for d, r in gold.items() if ((r.get("fields") or {}).get("발신명의") or {}).get("status") == "string_na"}

rows = {}
for s in (1, 2, 3):
    rows[s] = [json.loads(l) for l in (RES / f"layout_v3_seed{s}.jsonl").open(encoding="utf-8")]
seen = {r["doc_id"] for v in rows.values() for r in v}
stamped &= seen; normal = seen - stamped
print(f"평가 문서 {len(seen)}건 → 직인 {len(stamped)}건 · 정상 {len(normal)}건")

def cell(field, docs, key, old=False):
    per, ns = [], []
    for s, rs in rows.items():
        v = [(r["check"].get(key + "_ref") if old else r["faith"].get(key)) for r in rs if r["field"] == field and r["doc_id"] in docs]
        v = [x for x in v if x is not None]
        if v: per.append(st.mean(v)); ns.append(len(v))
    return (st.mean(per), st.pstdev(per) if len(per) > 1 else 0.0, round(st.mean(ns))) if per else (None, None, 0)

out = ["| 필드 | 집단 | 교환 구 | 교환 새 | δ=134 구 | δ=134 새 | n |", "|---|---|---:|---:|---:|---:|---:|"]
for f in FIELDS:
    for name, docs in (("직인", stamped), ("정상", normal)):
        so = cell(f, docs, "layout_swap", True); sn = cell(f, docs, "layout_swap")
        jo = cell(f, docs, "layout_jitter@134", True); jn = cell(f, docs, "layout_jitter@134")
        fm = lambda c: "—" if c[0] is None else f"{c[0]:.3f} ± {c[1]:.3f}"
        out.append(f"| {f} | {name} | {fm(so)} | {fm(sn)} | {fm(jo)} | {fm(jn)} | {sn[2]} |")
txt = "\n".join(out); print(txt)
(RES / "layout_v3_stamp_split.md").write_text("# 직인 겹침 대 정상 (layout_v3)\n\n" + txt + "\n", encoding="utf-8")
