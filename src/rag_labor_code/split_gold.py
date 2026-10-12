import csv
import io
import random
from collections import Counter
from pathlib import Path

PATH = Path("eval/gold.csv")

with PATH.open(encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f, delimiter=";")
    fieldnames = list(reader.fieldnames)
    rows = list(reader)

if "split" in fieldnames:
    raise SystemExit("В gold.csv уже есть столбец split, повторно делить нельзя.")

# проверяем данные до любой записи
bad = [
    r.get("id", "?")
    for r in rows
    if None in r or any(v is None for v in r.values()) or r["checked"] != "1"
]
if bad:
    raise SystemExit(f"Проблемные строки: {bad}. Файл не изменён.")

# делим отдельно по типам вопросов, чтобы в test попали и multi, и no_answer
rng = random.Random(42)
by_type: dict[str, list[dict]] = {}
for r in rows:
    by_type.setdefault(r["type"], []).append(r)
for group in by_type.values():
    rng.shuffle(group)
    cut = round(len(group) * 0.6)
    for i, r in enumerate(group):
        r["split"] = "dev" if i < cut else "test"

buf = io.StringIO()
writer = csv.DictWriter(buf, fieldnames=fieldnames + ["split"], delimiter=";", lineterminator="\n")
writer.writeheader()
writer.writerows(rows)
PATH.write_text(buf.getvalue(), encoding="utf-8", newline="")

print(Counter((r["split"], r["type"]) for r in rows))