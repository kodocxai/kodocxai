"""발신명의 골드 조정 — 레터헤드를 명의로 찍은 기록을 바로잡는다 (2026-08-14 저자 결정).

무엇이 문제였나
    시행문 맨 위의 「김 해 시」는 **기관명 레터헤드**이지 발신명의가 아니다. 발신명의는
    문서 하단(2면은 중상단)의 기관장 명의이며 「김해시장」처럼 직위를 포함한다(가이드 필드
    정의). 그런데 하단 명의가 없는 문서에서 주석자가 「없음」 대신 상단 레터헤드를 고른
    기록이 75건 있었다.

왜 고쳐야 하나
    레터헤드는 모든 문서 맨 위 같은 자리에 있어 필드로서 변별력이 없다. 그대로 두면 모델이
    「발신명의 = 왼쪽 위 로고」를 학습하고, 그 75건의 근거 정답 $R_k$ 가 로고 영역이 되어
    RQ2 타당성 평가가 오염된다.

세 갈래로 나눈다
    ① 서식 안쪽에 「김해시장(인)」이 있는 문서  →  그 스팬으로 고친다 (3건)
    ② 문서 어디에도 명의 흔적이 없는 문서      →  「없음」(absent) (50건)
    ③ 하단에 명의일 수도 있는 글자가 있는 문서 →  **자동 판정하지 않는다.** 주석자 재검토 (22건)

    ③을 자동으로 가르지 않는 이유: 본문 주소(「김해시 진영읍…」)와 직인에 가려진 명의
    (「김 해 시」 + 「장」 누락)가 문자열만으로 구분되지 않는다. 글자 크기도 판별에 쓸 수
    없다 — 확인된 발신명의 154건의 25%가 본문과 같은 크기였다.

IAA 는 이 조정의 영향을 받지 않는다. `_backup\\annotations_*.pre_adjudication.jsonl`
스냅숏으로 산출하며, `report_kodx.py` 가 그 파일을 우선한다.

사용
    python adjudicate_balsin.py              # 보고만 한다
    python adjudicate_balsin.py --apply      # 실제로 고친다
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path

from annotate import PILOT, span_of

BACKUP = PILOT / "_backup"
REVIEW = PILOT / "재검토_발신명의.txt"

TITLE = re.compile(r"장|수|청|서$")                    # 직위가 붙어 있으면 정상 기록
APPROV = re.compile(r"^(부시장|부군수|국장|과장|계장|시장|군수|담당|주사|기안|협조|심사|"
                    r"취급|보존|전결|경유)$")           # 결재란 어휘. 명의가 아니다
INLINE = re.compile(r"(김해|김\s*해).*(시장|군수)|시장\(인\)")
AGENCY = re.compile(r"^[김해시군읍면장수경상남도진영주촌한림생림대동상동장유칠산이북]+$")


def classify(rec: dict, doc: dict) -> tuple[str, list]:
    """('normal'|'inline'|'absent'|'review', 후보 어절 id)"""
    g = (rec.get("fields") or {}).get("발신명의") or {}
    if g.get("status") not in ("ok", "corrected", "string_na") or not g.get("bbox"):
        return "normal", []
    H = doc["image"]["height"]
    if g["bbox"][1] / H >= 0.3 or TITLE.search(g["value"].replace(" ", "")):
        return "normal", []

    words = doc["words"]
    hit = [w for w in words
           if INLINE.search(w["text"].replace(" ", ""))
           and not APPROV.match(w["text"].replace(" ", ""))]
    if hit:
        return "inline", [w["id"] for w in hit]

    low = [w for w in words
           if w["bbox"][1] / H > 0.55
           and AGENCY.match(w["text"].replace(" ", ""))
           and len(w["text"].replace(" ", "")) <= 4]
    byline: dict = {}
    for w in low:
        byline.setdefault(round(w["bbox"][1] / 40), []).append(w)
    if any(len(v) >= 2 for v in byline.values()):
        return "review", []
    return "absent", []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    docs = {d["doc_id"]: d for d in
            (json.loads(l) for l in
             (PILOT / "docs.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())}
    path = PILOT / "annotations_park.jsonl"
    recs = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]

    # 문서별로 한 번만 판정한다 (append-only 라 같은 doc_id 가 여러 줄이다)
    verdict: dict[str, tuple] = {}
    for r in recs:
        if r["doc_id"] not in verdict:
            verdict[r["doc_id"]] = classify(r, docs[r["doc_id"]])

    n_inline = n_absent = 0
    review = []
    for r in recs:
        kind, ids = verdict[r["doc_id"]]
        g = (r.get("fields") or {}).get("발신명의")
        if not g:
            continue
        if kind == "inline":
            d = docs[r["doc_id"]]
            text = {w["id"]: w["text"] for w in d["words"]}
            box = {w["id"]: w["bbox"] for w in d["words"]}
            g.update(span_of(sorted(ids), text, box))
            g["status"] = "ok"
            g["source"] = "adjudicated"
            n_inline += 1
        elif kind == "absent":
            g.update({"status": "absent", "value": "", "value_norm": "",
                      "word_ids": [], "bbox": None, "source": "adjudicated"})
            n_absent += 1

    for did, (kind, _) in verdict.items():
        if kind == "review":
            review.append(did)

    docs_inline = sum(1 for v in verdict.values() if v[0] == "inline")
    docs_absent = sum(1 for v in verdict.values() if v[0] == "absent")
    print(f"① 「김해시장(인)」 스팬으로 수정 : 문서 {docs_inline}건 (기록 {n_inline}줄)")
    print(f"② 「없음」으로 수정              : 문서 {docs_absent}건 (기록 {n_absent}줄)")
    print(f"③ 주석자 재검토 대상            : 문서 {len(review)}건 → {REVIEW.name}")

    if args.apply:
        BACKUP.mkdir(exist_ok=True)
        shutil.copy2(path, BACKUP / "annotations_park.pre_balsin_adjudication.jsonl")
        tmp = path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            for r in recs:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        tmp.replace(path)
        REVIEW.write_text("\n".join(sorted(review)) + "\n", encoding="utf-8")
        print(f"\n→ 적용 완료. 직전 판은 _backup\\annotations_park.pre_balsin_adjudication.jsonl")
        print(f"→ 재검토는:  python annotate.py --annotator park --only-ids ..\\_pilot\\{REVIEW.name}")
    else:
        print("\n실제로 고치려면 --apply 를 붙인다.")


if __name__ == "__main__":
    main()
