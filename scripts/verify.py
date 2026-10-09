# -*- coding: utf-8 -*-
"""Verify a reassembled KoDocXAI v1.0 benchmark against the public manifest.

Usage:
    python scripts/verify.py [--annotations annotations/] [--reassembled reassembled/]

Checks
  1. the public annotation files are byte-identical to the released version
     (SHA-256 recorded in ``annotations/manifest_public.json``);
  2. the reassembled directory (if present) contains every benchmark document,
     with every annotation span resolved against the reconstructed words.

Exit code 0 = all checks passed, 1 = at least one failure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", type=Path, default=HERE.parent / "annotations")
    ap.add_argument("--reassembled", type=Path, default=Path("reassembled"))
    a = ap.parse_args()

    failures = 0
    manifest = json.loads((a.annotations / "manifest_public.json").read_text(encoding="utf-8"))

    for name, meta in manifest["files"].items():
        p = a.annotations / name
        sha = hashlib.sha256(p.read_bytes()).hexdigest()
        n = sum(1 for line in p.open(encoding="utf-8") if line.strip())
        ok = sha == meta["sha256"] and n == meta["n_docs"]
        print(f"[{'OK' if ok else 'FAIL'}] {name}: {n} records, sha256 "
              f"{'matches' if sha == meta['sha256'] else 'DIFFERS'}")
        failures += 0 if ok else 1

    if not (a.reassembled / "docs.jsonl").exists():
        print(f"[SKIP] no reassembled output at {a.reassembled}/ "
              "(run scripts/reassemble.py first)")
        return 1 if failures else 0

    docs = {}
    for line in (a.reassembled / "docs.jsonl").open(encoding="utf-8"):
        d = json.loads(line)
        docs[d["doc_id"]] = {w["id"] for w in d["words"]}
    n_ann = 0
    bad_spans = 0
    primary = set()
    for line in (a.reassembled / "annotations.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        n_ann += 1
        if r["annotator"] == "A1":
            primary.add(r["doc_id"])
        wids = docs.get(r["doc_id"], set())
        for f, v in r["fields"].items():
            if any(i not in wids for i in v["word_ids"]):
                bad_spans += 1

    exp = manifest["n_primary_docs"]
    ok_docs = len(primary) == exp
    print(f"[{'OK' if ok_docs else 'FAIL'}] reassembled primary documents: "
          f"{len(primary)}/{exp}")
    print(f"[{'OK' if bad_spans == 0 else 'FAIL'}] unresolved spans: {bad_spans}")
    failures += (0 if ok_docs else 1) + (0 if bad_spans == 0 else 1)

    print("PASS" if failures == 0 else f"FAIL ({failures} problem(s))")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
