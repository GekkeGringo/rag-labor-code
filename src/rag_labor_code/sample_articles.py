import csv
import random
import sys
from pathlib import Path

from rag_labor_code.parser import parse

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
arts = parse(Path("data/raw/tk_rf.txt").read_text(encoding="utf-8"))

with open("eval/gold.csv", encoding="utf-8", newline="") as f:
    used = {n for row in csv.DictReader(f, delimiter=";") for n in row["articles"].split()}

pool = [a for a in arts if a.number not in used and sum(len(p) for p in a.paragraphs) > 300]
for a in random.Random(seed).sample(pool, 10):
    print(f"=== ст. {a.number}. {a.title}")
    print("\n".join(a.paragraphs)[:700])
    print()