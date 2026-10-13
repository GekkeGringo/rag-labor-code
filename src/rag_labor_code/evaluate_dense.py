from dataclasses import replace
from pathlib import Path

from rag_labor_code.chunker import chunk_articles
from rag_labor_code.dense import DenseIndex
from rag_labor_code.evaluate_retrieval import KS, evaluate, load_gold
from rag_labor_code.parser import parse
from rag_labor_code.retrieval import BM25Index

arts = parse(Path("data/raw/tk_rf.txt").read_text(encoding="utf-8"))
rows = load_gold("dev")
print(f"dev-вопросов с ответом: {len(rows)}")

for name, limit in [("статья целиком", None), ("до 1500 симв.", 1500), ("до 800 симв.", 800)]:
    chunks = [replace(c, text=f"{c.chapter}\n{c.text}") for c in chunk_articles(arts, limit)]
    for label, index in [("BM25 ", BM25Index(chunks)), ("dense", DenseIndex(chunks))]:
        rec, mrr = evaluate(index, rows)
        recall_str = "  ".join(f"R@{k}={rec[k]:.2f}" for k in KS)
        print(f"{name:16} {label}  {recall_str}  MRR={mrr:.2f}")