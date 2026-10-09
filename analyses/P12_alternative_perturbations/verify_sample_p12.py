import json, random
P = r"D:\AIHub데이터\_exp\P12_대체교란"
stored = json.load(open(P + r"\P12_표본100.json", encoding="utf-8"))
sample = set(stored["docs"]); assert len(sample) == 100
# seed1 jsonl 의 doc_id 최초 등장 순서 (표집 전 기계산분은 전체 문서 순서로 앞쪽에 몰려 있다)
order = []
seen = set()
for line in open(P + r"\P12_seed1.jsonl", encoding="utf-8"):
    if line.strip():
        d = json.loads(line)["doc_id"]
        if d not in seen:
            seen.add(d); order.append(d)
all_docs = [json.loads(l)["doc_id"] for l in open(r"D:\AIHub데이터\_pilot\docs.jsonl", encoding="utf-8") if l.strip()]
hits = []
for N in range(10, 71):
    done = order[:N]
    if not set(done) <= sample: break
    rest = [d for d in all_docs if d not in set(done)]
    rng = random.Random(20261002)
    extra = set(rng.sample(rest, 100 - N))
    if set(done) | extra == sample:
        hits.append(N)
print("RNG 재현이 성립하는 기계산 수 N:", hits)
print("N=34 검증:", "성립" if 34 in hits else "불성립")
print("jsonl 최초 등장 34건이 전부 표본 내:", set(order[:34]) <= sample)
