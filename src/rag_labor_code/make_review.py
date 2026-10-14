import json
from pathlib import Path

from rag_labor_code.parser import parse

arts = {a.number: a for a in parse(Path("data/raw/tk_rf.txt").read_text(encoding="utf-8"))}
ctx = json.loads(Path("data/contexts_dev.json").read_text(encoding="utf-8"))
ans = {
    x["id"]: x["answer"]
    for x in json.loads(Path("data/answers_dev.json").read_text(encoding="utf-8"))
}

lines = []
for item in ctx:
    if item["type"] == "no_answer":
        continue
    lines.append(f"## {item['id']}: {item['question']}\n")
    lines.append(f"**Ответ модели:** {ans[item['id']]}\n")
    for n in item["gold_articles"]:
        a = arts[n]
        text = "\n".join(a.paragraphs)
        lines.append(f"**Нужная ст. {n}. {a.title}:**\n\n{text[:1500]}\n")
    lines.append("---\n")

Path("data/review_dev.md").write_text("\n".join(lines), encoding="utf-8")

review = Path("eval/answer_review.csv")
if not review.exists():
    ids = [i["id"] for i in ctx if i["type"] != "no_answer"]
    review.write_text("id;verdict\n" + "\n".join(f"{i};" for i in ids) + "\n", encoding="utf-8")
    print("создана таблица оценок eval/answer_review.csv")
print("материал для чтения: data/review_dev.md")