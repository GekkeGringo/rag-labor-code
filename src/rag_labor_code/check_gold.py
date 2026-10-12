import csv
from pathlib import Path

from rag_labor_code.parser import parse

arts = {a.number: a for a in parse(Path("data/raw/tk_rf.txt").read_text(encoding="utf-8"))}

with open("eval/gold.csv", encoding="utf-8", newline="") as f:
    for row in csv.DictReader(f, delimiter=";"):
        print(f"{row['id']}: {row['question']}")
        if not row["articles"]:
            print("    (ответа в ТК быть не должно)")
        for num in row["articles"].split():
            a = arts.get(num)
            print(f"    -> ст. {num}: {a.title if a else 'НЕ НАЙДЕНА'}")