import random
from dataclasses import replace
from pathlib import Path

from rag_labor_code.chunker import chunk_articles
from rag_labor_code.dense import DenseIndex
from rag_labor_code.evaluate_retrieval import load_gold
from rag_labor_code.parser import parse
from rag_labor_code.retrieval import BM25Index


def per_question(index, rows, k=5):
    rrs, recalls = [], []
    for r in rows:
        gold = set(r["articles"].split())
        ranked = index.search_articles(r["question"], 10)
        pos = next((i + 1 for i, a in enumerate(ranked) if a in gold), None)
        rrs.append(1 / pos if pos else 0.0)
        recalls.append(len(gold & set(ranked[:k])) / len(gold))
    return rrs, recalls


def bootstrap_ci(diffs, n=10000, seed=0):
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(diffs, k=len(diffs))) / len(diffs) for _ in range(n))
    return sum(diffs) / len(diffs), means[int(0.025 * n)], means[int(0.975 * n)]


arts = parse(Path("data/raw/tk_rf.txt").read_text(encoding="utf-8"))
rows = load_gold("dev")
chunks = [replace(c, text=f"{c.chapter}\n{c.text}") for c in chunk_articles(arts, 1500)]

bm_rr, bm_rec = per_question(BM25Index(chunks), rows)
de_rr, de_rec = per_question(DenseIndex(chunks), rows)

for name, a, b in [("MRR", de_rr, bm_rr), ("Recall@5", de_rec, bm_rec)]:
    diffs = [x - y for x, y in zip(a, b)]
    mean, lo, hi = bootstrap_ci(diffs)
    print(f"{name}: dense - BM25 = {mean:+.2f}, 95% интервал [{lo:+.2f}; {hi:+.2f}]")

diffs = [x - y for x, y in zip(de_rr, bm_rr)]
print(f"Вопросов: dense лучше {sum(d > 0 for d in diffs)}, хуже {sum(d < 0 for d in diffs)}, равно {sum(d == 0 for d in diffs)}")