"""이미 단 주석에서 다른 안(案)의 스팬을 정답과 분리한다 (가이드 1-1, v1.3 소급 적용).

왜 필요한가
    공문서의 「1건 기안 다수 시행」 관행 때문에 한 장에 헤더 세트가 여러 벌 붙는다
    (파일럿 300건 중 69건 23.0%). 각 안은 서로 다른 수신자에게 나가는 별개의
    시행문이라, 한 필드로 이으면 어느 안의 정답도 아닌 문자열이 된다.

        idx 50 수신  '세무과장 내외 동장'        ← 3안의 수신 + 4안의 수신
        idx 50 제목  '건축허가에 따른 통보 건축허가에 따른 통보'

    $R_k$ 도 두 곳을 감싸는 사각형이 되어, 페이지 면적의 0.47% → 15.33% 로 부푼다.

무엇을 하는가
    정답은 **페이지 최상단 안** 하나로 남기고, 아래쪽 안의 스팬은 `secondary_spans` 로
    옮긴다. **버리지 않는다** — 평가에서 오답이 아니라 제외(ignore)로 다루기 위해
    위치가 필요하고, 주석자가 이미 클릭해 둔 덕에 공짜로 확보된 정보다.

사용
    python backfill_secondary.py              # 무엇이 바뀌는지 보고만 한다
    python backfill_secondary.py --apply      # 실제로 고친다 (원본은 _backup 에 남긴다)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path

from annotate import FIELDS, PILOT, split_secondary

BACKUP = PILOT / "_backup"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    docs = {d["doc_id"]: d for d in
            (json.loads(l) for l in
             (PILOT / "docs.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())}

    for path in sorted(PILOT.glob("annotations_*.jsonl")):
        recs = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        stat, log, n = Counter(), [], 0
        for r in recs:
            if r.get("secondary_spans"):        # 이미 분리된 기록은 건너뛴다
                stat["이미 분리됨"] += 1
                continue
            moved = split_secondary(r, docs[r["doc_id"]])
            if moved:
                n += 1
                for f, v in moved:
                    stat[f] += 1
                    log.append((r["index"], f, v))

        print(f"\n== {path.name} · {len(recs)}줄")
        print(f"   분리된 기록 {n}건 · 옮긴 스팬 {sum(stat[f] for f in FIELDS)}개")
        for f in FIELDS:
            if stat[f]:
                print(f"     {f:6s} {stat[f]}개")
        if stat["이미 분리됨"]:
            print(f"   이미 분리돼 건너뛴 기록 {stat['이미 분리됨']}건")
        for x in log[:12]:
            print(f"     idx {x[0]:3d} {x[1]:5s} → secondary  {x[2][:56]!r}")
        if len(log) > 12:
            print(f"     … 외 {len(log) - 12}개")

        if args.apply and n:
            BACKUP.mkdir(exist_ok=True)
            shutil.copy2(path, BACKUP / f"{path.stem}.v1.2_presecondary{path.suffix}")
            tmp = path.with_suffix(".tmp")
            with tmp.open("w", encoding="utf-8") as fh:
                for r in recs:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            tmp.replace(path)
            print(f"   → 적용 완료. 원본은 _backup\\{path.stem}.v1.2_presecondary{path.suffix}")

    if not args.apply:
        print("\n실제로 고치려면 --apply 를 붙인다.")


if __name__ == "__main__":
    main()
