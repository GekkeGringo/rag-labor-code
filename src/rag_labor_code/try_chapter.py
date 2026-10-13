from dataclasses import replace
from pathlib import Path

from rag_labor_code.chunker import chunk_articles
from rag_labor_code.evaluate_retrieval import KS, evaluate, load_gold
from rag_labor_code.parser import parse

from rag_labor_code.retrieval import BM25Index

arts = parse(Path("data/raw/tk_rf.txt").read_text(encoding="utf-8"))
rows = load_gold("dev")
base = chunk_articles(arts, 1500)
with_chapter = [replace(c, text=f"{c.chapter}\n{c.text}") for c in base]

for name, chunks in [("без главы", base), ("с названием главы", with_chapter)]:
    rec, mrr = evaluate(BM25Index(chunks), rows)
    print(f"{name:18}", "  ".join(f"R@{k}={rec[k]:.2f}" for k in KS), f" MRR={mrr:.2f}")