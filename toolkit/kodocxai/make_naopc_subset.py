"""NAOPC 부분표본 50건을 뽑는다.

NAOPC 는 상한·하한을 빔 서치로 찾느라 문서당 약 267초가 든다 — 나머지 지표 전부를 합친
것(28초)의 10배다. 300건 × 3시드면 67시간이라 현실적이지 않으므로 부분표본에서만 낸다.

**앞에서 N건을 자르면 안 된다.** 제시 순서가 무작위화돼 있어도 층 비율이 보장되지 않는다.
`docs.jsonl` 과 같은 18개 층(연도 3구간 × 제목 계열)에서 비례 배분한다.

선택은 결정적이다 — `sha256("kodx-naopc-v1" | doc_id)` 오름차순으로 각 층에서 앞에서부터
고른다. 같은 입력에 같은 표본이 나오므로 재현된다.

    python make_naopc_subset.py [--n 50]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

PILOT = Path(r"D:\AIHub데이터\_pilot")
SEED = "kodx-naopc-v1"


def norm(s: str | None) -> str:
    if not s:
        return ""
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", s))


def stratum(rec: dict) -> str:
    """report_kodx.py 의 층 정의와 같아야 한다."""
    year = int(rec["source"]["year"])
    band = "1994-96" if year <= 1996 else ("1997-99" if year <= 1999 else "2000+")
    title = norm((rec.get("fields", {}).get("제목") or {}).get("value_norm"))
    if not title:
        return f"{band} / 제목없음"
    for s in ("건축", "공장", "농지", "창업"):
        if s in title:
            return f"{band} / {s}"
    return f"{band} / 기타"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--out", type=Path, default=PILOT / "docs_naopc50.jsonl")
    a = ap.parse_args()

    docs = [json.loads(l) for l in
            (PILOT / "docs.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

    by: dict[str, list] = defaultdict(list)
    for d in docs:
        by[stratum(d)].append(d)
    for k in by:
        by[k].sort(key=lambda d: hashlib.sha256(
            f"{SEED}|{d['doc_id']}".encode()).hexdigest())

    # 비례 배분 — 내림한 뒤 남은 자리는 잔여가 큰 층부터 채운다
    total = len(docs)
    quota, rest = {}, {}
    for k, v in by.items():
        exact = len(v) * a.n / total
        quota[k] = int(exact)
        rest[k] = exact - quota[k]
    left = a.n - sum(quota.values())
    for k in sorted(rest, key=lambda k: -rest[k])[:left]:
        quota[k] += 1

    picked = [d for k in sorted(by) for d in by[k][:quota[k]]]
    # 원래 제시 순서를 유지한다 — 이어하기(resume) 로직이 순서에 기대지는 않지만
    # 사람이 진행 상황을 볼 때 헷갈리지 않는다
    order = {d["doc_id"]: i for i, d in enumerate(docs)}
    picked.sort(key=lambda d: order[d["doc_id"]])

    with a.out.open("w", encoding="utf-8") as f:
        for d in picked:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"NAOPC 부분표본 {len(picked)}건 → {a.out.name}")
    print(f"{'층':22s} 전체  표본")
    for k in sorted(by):
        print(f"  {k:20s} {len(by[k]):4d} {quota[k]:5d}")


if __name__ == "__main__":
    main()
