import json
from pathlib import Path

from rag_labor_code.parser import parse

arts = {a.number: a for a in parse(Path("data/raw/tk_rf.txt").read_text(encoding="utf-8"))}
ctx = json.loads(Path("data/contexts_test.json").read_text(encoding="utf-8"))
modes = ("rag", "oracle", "closed")
ans = {
    m: {
        x["id"]: x["answer"]
        for x in json.loads(Path(f"data/answers_test_{m}.json").read_text(encoding="utf-8"))
    }
    for m in modes
}

blocks = []
for item in ctx:
    i = item["id"]
    lines = [f"## {i} [{item['type']}]: {item['question']}\n"]
    for m in modes:
        if i in ans[m]:
            lines.append(f"**{m}:** {ans[m][i]}\n")
    for n in item["gold_articles"]:
        a = arts[n]
        text = "\n".join(a.paragraphs)
        lines.append(f"**Нужная ст. {n}. {a.title}:** {text[:1200]}\n")
    lines.append("---\n")
    blocks.append("\n".join(lines))

half = len(blocks) // 2
Path("data/review_test_1.md").write_text("\n".join(blocks[:half]), encoding="utf-8")
Path("data/review_test_2.md").write_text("\n".join(blocks[half:]), encoding="utf-8")
print(f"вопросов: {len(blocks)}; файлы data/review_test_1.md и data/review_test_2.md")