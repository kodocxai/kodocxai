import json, random
done_docs = []
seen = set()
for line in open(r"D:\AIHub데이터\_exp\P12_대체교란\P12_seed1.jsonl", encoding="utf-8"):
    if line.strip():
        d = json.loads(line)["doc_id"]
        if d not in seen:
            seen.add(d); done_docs.append(d)
all_docs = []
for line in open(r"D:\AIHub데이터\_pilot\docs.jsonl", encoding="utf-8"):
    if line.strip():
        all_docs.append(json.loads(line)["doc_id"])
rest = [d for d in all_docs if d not in seen]
rng = random.Random(20261002)
extra = rng.sample(rest, 100 - len(done_docs))
sample = sorted(done_docs + extra)
json.dump({"n": len(sample), "rule": "기계산 25 + RNG 20261002 표집 75, 시드 공통", "docs": sample},
          open(r"D:\AIHub데이터\_exp\P12_대체교란\P12_표본100.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("표본", len(sample), "건 (기계산", len(done_docs), "+ 신규", len(extra), ")")
