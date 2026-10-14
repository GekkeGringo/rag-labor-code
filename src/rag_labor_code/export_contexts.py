import csv
import json
import sys
from dataclasses import replace
from pathlib import Path

from rag_labor_code.chunker import chunk_articles
from rag_labor_code.dense import DenseIndex
from rag_labor_code.parser import parse

split = sys.argv[1] if len(sys.argv) > 1 else "dev"
arts = parse(Path("data/raw/tk_rf.txt").read_text(encoding="utf-8"))
chunks = [replace(c, text=f"{c.chapter}\n{c.text}") for c in chunk_articles(arts, 1500)]
index = DenseIndex(chunks)

with open("eval/gold.csv", encoding="utf-8", newline="") as f:
    rows = [r for r in csv.DictReader(f, delimiter=";") if r["split"] == split]

out = []
for r in rows:
    found = index.search_chunks(r["question"], 5)
    out.append(
        {
            "id": r["id"],
            "question": r["question"],
            "type": r["type"],
            "gold_articles": r["articles"].split(),
            "contexts": [{"article": c.article, "text": c.text} for c in found],
        }
    )

path = Path(f"data/contexts_{split}.json")
path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"{split}: {len(out)} вопросов -> {path}")
print("пример:", out[0]["question"], "->", [c["article"] for c in out[0]["contexts"]])