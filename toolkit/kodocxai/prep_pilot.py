"""주석 파일럿 세트 구성.

인.허가 시행문에서 연도 × 제목 계열로 층화 추출하고, 원천 zip에서 해당 이미지를
그대로(재인코딩 없이) 꺼내 `_pilot\\` 에 떨군다.

산출물
  _pilot/docs.jsonl     어댑터 공통 스키마 (words / lines / fields)
  _pilot/images/<id>.jpg 원본 해상도 JPEG
  _pilot/manifest.json   층화 내역과 구성 근거

표준 라이브러리만 사용한다.

사용 예
  python prep_pilot.py --n 300
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from aihub_adapter import entry_name, iter_docs   # noqa: E402

BASE = Path(r"D:\AIHub데이터\공공행정문서 OCR")
LABEL_ZIP = BASE / "Training" / "[라벨]train.zip"
SRC_ZIP = BASE / "Training" / "[원천]train1.zip"
OUT = Path(r"D:\AIHub데이터\_pilot")

# 제목 계열 — 인허가_제목빈도_v2.csv 집계 기준 (공장 45.9 / 건축 16.5 / 창업 12.2 / 농지 7.3)
FAMILIES = ("공장", "건축", "창업", "농지")


def family_of(doc) -> str:
    t = (doc.fields.get("제목") or {}).get("value", "")
    if not t:
        return "제목없음"          # 2면 이후·별지. 전체의 약 8%
    for f in FAMILIES:
        if f in t:
            return f
    return "기타"


def year_bucket(doc) -> str:
    """연도는 1994~2003에 92%가 몰려 있다. 3구간으로 나눈다."""
    y = doc.source.get("year")
    try:
        y = int(y)
    except (TypeError, ValueError):
        return "미상"
    if y <= 1996:
        return "1994-96"
    if y <= 1999:
        return "1997-99"
    return "2000+"


def main() -> int:
    ap = argparse.ArgumentParser(description="주석 파일럿 세트 구성")
    ap.add_argument("--n", type=int, default=300, help="추출할 문서 수")
    ap.add_argument("--pool", type=int, default=3000, help="층화 전 후보 풀 크기")
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "images").mkdir(exist_ok=True)

    # 1) 후보 풀을 균등 간격으로 훑는다. 전수 파싱은 불필요하고 느리다.
    print(f"후보 {a.pool}건 수집 중…", flush=True)
    strata: dict[tuple, list] = defaultdict(list)
    total = 0
    with zipfile.ZipFile(LABEL_ZIP) as z:
        n_all = sum(1 for i in z.infolist()
                    if not i.is_dir() and "/인.허가/" in entry_name(i)
                    and entry_name(i).lower().endswith(".json"))
    step = max(1, n_all // a.pool)

    for doc in iter_docs(LABEL_ZIP, "/인.허가/", a.pool, step):
        total += 1
        strata[(year_bucket(doc), family_of(doc))].append(doc)
        if total % 500 == 0:
            print(f"  {total}건…", flush=True)

    print(f"후보 {total}건 / 층 {len(strata)}개 (전체 {n_all}건, step={step})")

    # 2) 층 크기에 비례 배분하되 각 층 최소 1건은 보장한다.
    keys = sorted(strata, key=lambda k: -len(strata[k]))
    picked = []
    remaining = a.n
    for i, k in enumerate(keys):
        left_strata = len(keys) - i
        share = max(1, round(a.n * len(strata[k]) / total))
        share = min(share, remaining - (left_strata - 1), len(strata[k]))
        if share <= 0:
            continue
        # 층 안에서도 균등 간격으로 뽑아 특정 기관·연도 쏠림을 줄인다.
        # stride 결과를 share로 잘라야 한다. 안 자르면 큰 층이 배분량을 초과해
        # 뒤쪽 층의 몫을 잠식한다.
        pool = strata[k]
        stride = max(1, len(pool) // share)
        picked.extend(list(pool[i2] for i2 in range(0, len(pool), stride))[:share])
        remaining = a.n - len(picked)
        if remaining <= 0:
            break

    print(f"층화 추출 {len(picked)}건")

    # 3) 원천 zip에서 이미지를 그대로 꺼낸다. 재인코딩하지 않는다(Pillow 불필요).
    want = {d.image["file_name"]: d for d in picked if d.image.get("file_name")}
    print(f"원천 이미지 {len(want)}건 추출 중…", flush=True)
    found = 0
    with zipfile.ZipFile(SRC_ZIP) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            name = Path(entry_name(info)).name
            if name not in want:
                continue
            (OUT / "images" / name).write_bytes(z.read(info))
            found += 1
            if found % 50 == 0:
                print(f"  {found}건…", flush=True)

    missing = [n for n in want if not (OUT / "images" / n).exists()]
    if missing:
        print(f"  ! 원천 없음 {len(missing)}건 — 파일럿에서 제외한다")

    # 4) 이미지가 실제로 있는 것만 남긴다
    final = [d for d in picked
             if d.image.get("file_name") and (OUT / "images" / d.image["file_name"]).exists()]

    with (OUT / "docs.jsonl").open("w", encoding="utf-8") as f:
        for d in final:
            f.write(d.to_json() + "\n")

    dist: dict[str, int] = defaultdict(int)
    fields_n: dict[str, int] = defaultdict(int)
    for d in final:
        dist[f"{year_bucket(d)} / {family_of(d)}"] += 1
        for k in d.fields:
            fields_n[k] += 1

    manifest = {
        "n_docs": len(final),
        "source": {
            "label_zip": str(LABEL_ZIP), "src_zip": str(SRC_ZIP),
            "category": "인.허가", "population": n_all, "pool": total, "pool_step": step,
        },
        "strata": dict(sorted(dist.items())),
        "auto_fields": dict(sorted(fields_n.items(), key=lambda kv: -kv[1])),
        "note": "자동 추출 필드는 검수 전 초안이다. 정답이 아니다.",
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n파일럿 {len(final)}건 → {OUT}")
    print("\n--- 층 분포 ---")
    for k, v in sorted(dist.items()):
        print(f"  {k:<22} {v:>4}")
    print("\n--- 자동 추출 필드 (검수 전) ---")
    for k, v in sorted(fields_n.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<8} {v:>4} / {len(final)}  ({100*v/max(1,len(final)):.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
