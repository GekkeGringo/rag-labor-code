import json
import re
from pathlib import Path

ctx = json.loads(Path("data/contexts_dev.json").read_text(encoding="utf-8"))
ans = {
    x["id"]: x["answer"]
    for x in json.loads(Path("data/answers_dev.json").read_text(encoding="utf-8"))
}
CITE = re.compile(r"ст\.\s*(\d+(?:\.\d+)*(?:-\d+)?)")

n_ans = n_refused = n_ctx_ok = n_cited_ok = n_answered = n_bad_cite = 0
n_no = n_no_refused = 0

for item in ctx:
    a = ans[item["id"]]
    refused = "ответа нет" in a.lower()
    cited = set(CITE.findall(a))
    in_ctx = [c["article"] for c in item["contexts"]]
    gold = set(item["gold_articles"])
    if item["type"] == "no_answer":
        n_no += 1
        n_no_refused += refused
        print(f"{item['id']} без ответа | отказ={'да' if refused else 'НЕТ'}")
        continue
    n_ans += 1
    n_refused += refused
    ctx_ok = gold <= set(in_ctx)
    n_ctx_ok += ctx_ok
    line = (
        f"{item['id']} нужна {sorted(gold)} | в контексте: {'да' if ctx_ok else 'НЕТ'}"
        f" | отказ: {'ДА' if refused else 'нет'} | цитирует: {sorted(cited)}"
    )
    if not refused:
        n_answered += 1
        n_cited_ok += gold <= cited
        n_bad_cite += bool(cited - set(in_ctx))
    if not ctx_ok:
        line += f" | контекст: {in_ctx}"
    print(line)

print()
print(f"вопросов с ответом: {n_ans}; нужные статьи в контексте: {n_ctx_ok}")
print(f"ложных отказов: {n_refused} из {n_ans}")
print(f"из {n_answered} ответов нужные статьи процитированы в {n_cited_ok}, "
      f"в {n_bad_cite} есть ссылка на статью вне контекста")
print(f"вопросов без ответа: {n_no}; правильных отказов: {n_no_refused}")