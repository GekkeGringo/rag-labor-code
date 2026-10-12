import csv
from pathlib import Path

from rag_labor_code.chunker import chunk_articles
from rag_labor_code.parser import parse
from rag_labor_code.retrieval import BM25Index

KS = (1, 3, 5, 10)


def load_gold(split: str) -> list[dict]:
    with open("eval/gold.csv", encoding="utf-8", newline="") as f:
        return [r for r in csv.DictReader(f, delimiter=";") if r["split"] == split and r["articles"]]


def evaluate(index, rows):
    recalls = {k: [] for k in KS}
    rr = []
    for r in rows:
        gold = set(r["articles"].split())
        ranked = index.search_articles(r["question"], max(KS))
        for k in KS:
            recalls[k].append(len(gold & set(ranked[:k])) / len(gold))
        pos = next((i + 1 for i, a in enumerate(ranked) if a in gold), None)
        rr.append(1 / pos if pos else 0.0)
    return {k: sum(v) / len(v) for k, v in recalls.items()}, sum(rr) / len(rr)


if __name__ == "__main__":
    arts = parse(Path("data/raw/tk_rf.txt").read_text(encoding="utf-8"))
    rows = load_gold("dev")
    print(f"dev-вопросов с ответом: {len(rows)}")
    for name, limit in [("статья целиком", None), ("до 1500 симв.", 1500), ("до 800 симв.", 800)]:
        rec, mrr = evaluate(BM25Index(chunk_articles(arts, limit)), rows)
        recall_str = "  ".join(f"R@{k}={rec[k]:.2f}" for k in KS)
        print(f"{name:16} {recall_str}  MRR={mrr:.2f}")