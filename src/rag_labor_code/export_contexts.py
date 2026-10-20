import csv
import json
import sys
from pathlib import Path

from rag_labor_code.dense import DenseIndex
from rag_labor_code.pipeline import INDEX_DIR

split = sys.argv[1] if len(sys.argv) > 1 else "dev"
version = (INDEX_DIR / "CURRENT.txt").read_text(encoding="utf-8").strip()
index = DenseIndex.load(INDEX_DIR / version)

by_article: dict[str, list] = {}
for c in index.chunks:
    by_article.setdefault(c.article, []).append(c)


def cap(chunks, limit=8000):
    result, total = [], 0
    for c in chunks:
        if total + len(c.text) > limit and result:
            break
        result.append(c)
        total += len(c.text)
    return result


def pack(chunks):
    return [{"article": c.article, "text": c.text} for c in chunks]


with open("eval/gold.csv", encoding="utf-8", newline="") as f:
    rows = [r for r in csv.DictReader(f, delimiter=";") if r["split"] == split]

out = []
for r in rows:
    gold = r["articles"].split()
    item = {
        "id": r["id"],
        "question": r["question"],
        "type": r["type"],
        "gold_articles": gold,
        "contexts": pack(index.search_chunks(r["question"], 5)),
    }
    if gold:
        item["oracle_contexts"] = pack(cap([c for a in gold for c in by_article[a]]))
    out.append(item)

path = Path(f"data/contexts_{split}.json")
path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"{split}: {len(out)} вопросов, индекс {version} -> {path}")