import random
from pathlib import Path

from rag_labor_code.dense import DenseIndex
from rag_labor_code.evaluate_retrieval import KS, evaluate, load_gold
from rag_labor_code.pipeline import INDEX_DIR
from rag_labor_code.retrieval import BM25Index

OUT = Path("eval/test_retrieval_result.txt")
if OUT.exists():
    raise SystemExit("test уже открывали, повторный запуск запрещён: см. " + str(OUT))


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


version = (INDEX_DIR / "CURRENT.txt").read_text(encoding="utf-8").strip()
dense = DenseIndex.load(INDEX_DIR / version)
bm25 = BM25Index(dense.chunks)
rows = load_gold("test")

lines = [f"индекс {version}, test-вопросов с ответом: {len(rows)}"]
for name, idx in [("BM25 ", bm25), ("dense", dense)]:
    rec, mrr = evaluate(idx, rows)
    lines.append(f"{name} " + "  ".join(f"R@{k}={rec[k]:.2f}" for k in KS) + f"  MRR={mrr:.2f}")

bm_rr, bm_rec = per_question(bm25, rows)
de_rr, de_rec = per_question(dense, rows)
for label, a, b in [("MRR", de_rr, bm_rr), ("Recall@5", de_rec, bm_rec)]:
    diffs = [x - y for x, y in zip(a, b)]
    mean, lo, hi = bootstrap_ci(diffs)
    lines.append(f"{label}: dense - BM25 = {mean:+.2f}, 95% интервал [{lo:+.2f}; {hi:+.2f}]")
d = [x - y for x, y in zip(de_rr, bm_rr)]
lines.append(f"Вопросов: dense лучше {sum(v > 0 for v in d)}, хуже {sum(v < 0 for v in d)}, равно {sum(v == 0 for v in d)}")

text = "\n".join(lines)
print(text)
OUT.write_text(text + "\n", encoding="utf-8")