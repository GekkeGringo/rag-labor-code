from pathlib import Path

from rag_labor_code.chunker import chunk_articles
from rag_labor_code.evaluate_retrieval import load_gold
from rag_labor_code.parser import parse
from rag_labor_code.retrieval import BM25Index

arts = parse(Path("data/raw/tk_rf.txt").read_text(encoding="utf-8"))
titles = {a.number: a.title[:70] for a in arts}
index = BM25Index(chunk_articles(arts, 1500))

for r in load_gold("dev"):
    gold = r["articles"].split()
    if not gold:
        continue
    ranked = index.search_articles(r["question"], 10)
    if not set(gold) & set(ranked[:3]):
        print(f"{r['id']}: {r['question']}")
        for g in gold:
            place = ranked.index(g) + 1 if g in ranked else ">10"
            print(f"   нужна ст. {g} ({titles[g]}), место в выдаче: {place}")
        for a in ranked[:3]:
            print(f"   нашли ст. {a}: {titles[a]}")