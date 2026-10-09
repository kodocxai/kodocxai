# -*- coding: utf-8 -*-
"""Reassemble the KoDocXAI v1.0 benchmark from your own AI-Hub download.

The public annotation layer in ``annotations/`` contains only document
identifiers, field types, status labels, annotation-ID index spans, label-file
paths, and integrity hashes.  No source images, no transcribed text, and no
coordinates are distributed.  To reconstruct the benchmark you must download
the source dataset yourself:

  1. Apply for and download "공공행정문서 OCR" (Public Administrative
     Documents OCR, dataSetSn=88) at https://aihub.or.kr (application is
     currently limited to residents of South Korea; see the paper, Section 5.5).
     Only ``Training/[라벨]train.zip`` is required for labels; document images
     used by the image modality come from ``Training/[원천]train*.zip``.
  2. Run this script:

       python scripts/reassemble.py --label-zip path/to/[라벨]train.zip \\
              [--src-zip path/to/[원천]train1.zip ...] [--out reassembled]

  3. Integrity check only (no output written):

       python scripts/reassemble.py --label-zip path/to/[라벨]train.zip --verify-only

Per the AI-Hub terms of use, downloaded data must not be passed to third
parties; every user (including co-researchers) must apply individually.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "toolkit"))
try:
    from kodocxai.aihub_adapter import (  # noqa: E402
        build_lines, entry_name, extract_fields, normalize)
except ImportError as e:  # pragma: no cover
    sys.exit(f"error: cannot import the bundled toolkit ({e}). "
             "Run this script from the repository checkout.")

FIELDS = ["제목", "수신", "문서번호", "시행일자", "발신명의"]


def load_layer(ann_dir: Path) -> list[dict]:
    records = []
    for name in ("kodocxai_v1.0.jsonl", "kodocxai_v1.0_double.jsonl"):
        p = ann_dir / name
        if not p.exists():
            sys.exit(f"error: annotation file not found: {p}")
        for line in p.open(encoding="utf-8"):
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def index_zip(zpath: Path) -> tuple[zipfile.ZipFile, dict]:
    if not zpath.exists():
        sys.exit(f"error: zip not found: {zpath}\n"
                 "Download Training/[라벨]train.zip from AI-Hub first "
                 "(you only need the 인.허가 portion, not the full 797 GB).")
    z = zipfile.ZipFile(zpath)
    # AI-Hub zips store Korean entry names in CP949; decode via cp437 round-trip.
    idx = {entry_name(i): i for i in z.infolist() if not i.is_dir()}
    return z, idx


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--label-zip", required=True, type=Path)
    ap.add_argument("--src-zip", action="append", type=Path, default=[],
                    help="[원천]train*.zip archives holding the document images "
                         "(optional; repeatable)")
    ap.add_argument("--annotations", type=Path, default=HERE.parent / "annotations")
    ap.add_argument("--out", type=Path, default=Path("reassembled"))
    ap.add_argument("--verify-only", action="store_true",
                    help="check label-file hashes against the annotation layer "
                         "and exit (writes nothing)")
    a = ap.parse_args()

    layer = load_layer(a.annotations)
    z, idx = index_zip(a.label_zip)

    n_ok = n_missing = n_mismatch = 0
    docs_out: dict[str, dict] = {}
    ann_out: list[dict] = []
    for rec in layer:
        doc_id = rec["doc_id"]
        entry = rec["source"]["label_entry"]
        info = idx.get(entry)
        if info is None:
            print(f"MISSING  {doc_id}: entry not in zip: {entry}")
            n_missing += 1
            continue
        raw = z.read(info)
        sha = hashlib.sha256(raw).hexdigest()
        if sha != rec["source"]["label_sha256"]:
            print(f"MISMATCH {doc_id}: label file hash differs from the "
                  f"annotated version (dataset revision changed?)")
            n_mismatch += 1
            continue
        n_ok += 1
        if a.verify_only:
            continue

        if doc_id not in docs_out:
            # Same pipeline as the original benchmark build (iter_docs):
            # normalize -> line grouping -> rule-based auto fields.
            doc = normalize(json.loads(raw.decode("utf-8")), entry)
            doc.lines = build_lines(doc.words)
            doc.fields = extract_fields(doc)
            docs_out[doc_id] = json.loads(doc.to_json())
        d = docs_out[doc_id]
        # Some AI-Hub label files carry duplicate annotation ``id`` values.
        # References are therefore (id, n-th occurrence); records without an
        # ``annotation_occ`` key mean "first occurrence" throughout.
        by_ref: dict[tuple, dict] = {}
        seen: dict[int, int] = {}
        for w in d["words"]:
            occ = seen.get(w["orig_id"], 0)
            by_ref[(w["orig_id"], occ)] = w
            seen[w["orig_id"]] = occ + 1

        def to_refs(span: dict) -> list[dict]:
            ids = span["annotation_ids"]
            occ = span.get("annotation_occ", [0] * len(ids))
            missing = [i for i, o in zip(ids, occ) if (i, o) not in by_ref]
            if missing:
                sys.exit(f"error: {doc_id}: annotation ids {missing} not found "
                         "in the label file (unexpected; please report)")
            return [by_ref[(i, o)] for i, o in zip(ids, occ)]

        fields = {}
        for f in FIELDS:
            v = rec["fields"][f]
            ws = to_refs(v)
            fields[f] = {
                "status": v["status"],
                "word_ids": [w["id"] for w in ws],
                # Text is joined from YOUR OWN AI-Hub copy, not distributed by us.
                "value": " ".join(w["text"] for w in ws),
            }
        ann_out.append({
            "doc_id": doc_id,
            "annotator": rec["annotator"],
            "guideline": rec["guideline"],
            "fields": fields,
            "secondary_spans": [{"field": sp["field"],
                                 "word_ids": [w["id"] for w in to_refs(sp)]}
                                for sp in rec.get("secondary_spans", [])],
        })

    print(f"\nlabel files: {n_ok} verified, {n_missing} missing, {n_mismatch} hash mismatch")
    if a.verify_only:
        return 0 if (n_missing == 0 and n_mismatch == 0) else 1
    if n_missing or n_mismatch:
        print("aborting: fix the issues above and re-run.")
        return 1

    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out / "docs.jsonl").open("w", encoding="utf-8") as f:
        for doc_id in sorted(docs_out):
            f.write(json.dumps(docs_out[doc_id], ensure_ascii=False) + "\n")
    with (a.out / "annotations.jsonl").open("w", encoding="utf-8") as f:
        for r in ann_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if a.src_zip:
        wanted = {docs_out[d]["image"]["file_name"] for d in docs_out}
        img_dir = a.out / "images"
        img_dir.mkdir(exist_ok=True)
        found = 0
        for zp in a.src_zip:
            with zipfile.ZipFile(zp) as sz:
                for info in sz.infolist():
                    if info.is_dir():
                        continue
                    name = Path(entry_name(info)).name
                    if name in wanted and not (img_dir / name).exists():
                        (img_dir / name).write_bytes(sz.read(info))
                        found += 1
        print(f"images: {found}/{len(wanted)} extracted")
        if found < len(wanted):
            print("note: remaining images are in other [원천]train*.zip parts; "
                  "pass them with additional --src-zip arguments.")

    print(f"done: benchmark written to {a.out}/ "
          f"({len(docs_out)} documents, {len(ann_out)} annotation records)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
