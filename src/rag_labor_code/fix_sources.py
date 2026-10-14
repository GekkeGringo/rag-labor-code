import csv
import io
from collections import Counter
from pathlib import Path

# если все 24 оценки мои, замените строку ниже на: HUMAN = set()
HUMAN = {"q05", "q11", "q17", "q28", "q31", "q35", "q40", "q42"}
PATH = Path("eval/answer_review.csv")

with PATH.open(encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f, delimiter=";"))
for r in rows:
    r["source"] = "human" if r["id"] in HUMAN else "claude"

buf = io.StringIO()
writer = csv.DictWriter(buf, fieldnames=["id", "verdict", "source"], delimiter=";", lineterminator="\n")
writer.writeheader()
writer.writerows(rows)
PATH.write_text(buf.getvalue(), encoding="utf-8", newline="")
print(dict(Counter(r["source"] for r in rows)))