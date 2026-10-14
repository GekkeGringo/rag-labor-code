import csv
import io
import math
from collections import Counter
from pathlib import Path

CLAUDE = {
    "q04": "ok", "q10": "ok", "q12": "ok", "q13": "ok",
    "q26": "ok", "q27": "ok", "q29": "ok", "q30": "partial",
    "q32": "ok", "q33": "wrong", "q34": "partial", "q36": "ok",
    "q37": "ok", "q38": "ok", "q41": "ok", "q43": "ok",
}
VALID = {"ok", "partial", "wrong"}
PATH = Path("eval/answer_review.csv")

with PATH.open(encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f, delimiter=";"))

if "source" in rows[0]:
    raise SystemExit("Столбец source уже есть, повторно заполнять нельзя.")

# сначала проверяем всё, потом пишем
for r in rows:
    v = (r["verdict"] or "").strip()
    if v:
        if v not in VALID:
            raise SystemExit(f"{r['id']}: неизвестная оценка '{v}'. Файл не изменён.")
        r["verdict"], r["source"] = v, "human"
    elif r["id"] in CLAUDE:
        r["verdict"], r["source"] = CLAUDE[r["id"]], "claude"
    else:
        raise SystemExit(f"{r['id']}: нет оценки. Файл не изменён.")

buf = io.StringIO()
writer = csv.DictWriter(buf, fieldnames=["id", "verdict", "source"], delimiter=";", lineterminator="\n")
writer.writeheader()
writer.writerows(rows)
PATH.write_text(buf.getvalue(), encoding="utf-8", newline="")


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - m, c + m


n = len(rows)
cnt = Counter(r["verdict"] for r in rows)
print("источники:", dict(Counter(r["source"] for r in rows)))
print("оценки:", dict(cnt))
lo, hi = wilson(cnt["ok"], n)
print(f"полностью верно: {cnt['ok']} из {n} ({cnt['ok'] / n:.0%}), 95% интервал [{lo:.0%}; {hi:.0%}]")
print(f"верно или частично: {cnt['ok'] + cnt['partial']} из {n} ({(cnt['ok'] + cnt['partial']) / n:.0%})")