"""이미 단 주석의 스팬에서 앞뒤 구두점 어절을 떼어낸다 (가이드 5-4, v1.2 소급 적용).

왜 필요한가
    어댑터는 「수 신 : 김해시…」의 콜론을 값 문자열에서만 빼고 `word_ids`·`bbox` 에는
    남긴다. 그래서 주석 화면에서 콜론을 빼든 두든 똑같아 보이고, 실제로 v1.1 기간의
    기록은 뺀 것과 남긴 것이 섞였다. 스팬은 근거 정답 $R_k$ 이고 IAA 스팬 일치율의
    입력이므로, 섞여 있으면 두 수치가 주석자의 변덕을 반영하게 된다.

무엇을 하지 않는가
    - `docs.jsonl` 과 어댑터는 건드리지 않는다 (어댑터 기준선의 기준점)
    - **스팬 안쪽 구두점은 손대지 않는다.** 「공영16000 - 135」의 하이픈,
      「1997. 8 . 13」의 점은 값의 일부다
    - **시행일자 말미의 마침표도 남긴다.** 「96. 8. 16 .」의 끝 점은 날짜 표기의 일부다
    - `status`·`source`·`ts`·`dwell_ms` 는 그대로 둔다. 주석자가 실제로 한 행위의 기록이다

사용
    python backfill_edgepunct.py              # 무엇이 바뀌는지 보고만 한다
    python backfill_edgepunct.py --apply      # 실제로 고친다 (원본은 _backup 에 남긴다)
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

from annotate import FIELDS, PILOT, _EDGE_PUNCT, span_of, trim_ids

BACKUP = PILOT / "_backup"


def load_docs() -> dict:
    return {d["doc_id"]: d for d in
            (json.loads(l) for l in
             (PILOT / "docs.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())}


def fix_record(rec: dict, doc: dict, stat: Counter, log: list) -> bool:
    text = {w["id"]: w.get("text", "") for w in doc["words"]}
    box = {w["id"]: w["bbox"] for w in doc["words"]}
    changed = False
    for f in FIELDS:
        g = (rec.get("fields") or {}).get(f)
        if not g:
            continue
        ids = list(g.get("word_ids") or [])
        if not ids:
            continue
        keep = trim_ids(f, ids, text)
        # 안쪽 구두점은 세기만 하고 두는지 확인한다 (회귀 방지)
        inner = sum(1 for i in ids[1:-1] if _EDGE_PUNCT.fullmatch(text.get(i, "\0") or "\0"))
        stat["안쪽 유지"] += inner
        if keep == ids:
            stat["변경 없음"] += 1
            continue
        if not keep:
            stat["구두점뿐 — 건너뜀"] += 1
            log.append((rec["index"], f, "구두점뿐이라 두었다", g.get("value")))
            continue
        before = g.get("value")
        g.update(span_of(keep, text, box))
        stat["스팬 정리"] += 1
        if before != g["value"]:
            stat["값 문자열도 바뀜"] += 1
            log.append((rec["index"], f, repr(before), repr(g["value"])))
        changed = True
    return changed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    docs = load_docs()
    for path in sorted(PILOT.glob("annotations_*.jsonl")):
        recs = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        stat, log, n = Counter(), [], 0
        for r in recs:
            if fix_record(r, docs[r["doc_id"]], stat, log):
                n += 1
        print(f"\n== {path.name} · {len(recs)}건")
        for k, v in stat.items():
            print(f"   {k:16s} {v}")
        print(f"   기록 {n}건이 바뀐다")
        for x in log[:12]:
            print("     idx", x[0], x[1], x[2], "->", x[3])
        if len(log) > 12:
            print(f"     … 외 {len(log) - 12}건")

        if args.apply and n:
            BACKUP.mkdir(exist_ok=True)
            shutil.copy2(path, BACKUP / f"{path.stem}.v1.1_edgepunct{path.suffix}")
            tmp = path.with_suffix(".tmp")
            with tmp.open("w", encoding="utf-8") as fh:
                for r in recs:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                fh.flush()
                import os
                os.fsync(fh.fileno())
            tmp.replace(path)
            print(f"   → 적용 완료. 원본은 _backup\\{path.stem}.v1.1_edgepunct{path.suffix}")

    if not args.apply:
        print("\n실제로 고치려면 --apply 를 붙인다.")


if __name__ == "__main__":
    main()
