import json
import re
import sys
from pathlib import Path

mode = sys.argv[1]
ctx = json.loads(Path("data/contexts_test.json").read_text(encoding="utf-8"))
ans = {
    x["id"]: x["answer"]
    for x in json.loads(Path(f"data/answers_test_{mode}.json").read_text(encoding="utf-8"))
}
CITE = re.compile(r"ст\.\s*(\d+(?:\.\d+)*(?:-\d+)?)")


def is_refusal(a: str) -> bool:
    low = a.lower()
    if mode == "closed":
        return bool(re.search(r"ответа[^.]{0,60}\bнет\b|не знаю", low))
    return not CITE.search(a) and bool(re.search(r"\bнет\b", low))


n_ans = n_ref = n_cit_gold = n_cit_any = n_cit_extra = 0
n_no = n_no_ref = 0
print("режим:", mode)
for item in ctx:
    if item["id"] not in ans:
        continue
    a = ans[item["id"]]
    ref = is_refusal(a)
    cited = set(CITE.findall(a))
    gold = set(item["gold_articles"])
    if item["type"] == "no_answer":
        n_no += 1
        n_no_ref += ref
        print(f"{item['id']} без ответа | отказ={'да' if ref else 'НЕТ'} | {a[:140]!r}")
        continue
    n_ans += 1
    n_ref += ref
    if not ref:
        n_cit_any += bool(cited)
        n_cit_gold += gold <= cited
        n_cit_extra += bool(cited - gold)
    print(f"{item['id']} нужна {sorted(gold)} | отказ: {'ДА' if ref else 'нет'} | цитирует: {sorted(cited)}")

print()
print(f"вопросов с ответом: {n_ans}; отказов: {n_ref}")
print(
    f"из {n_ans - n_ref} ответов: ссылка на статью есть в {n_cit_any}, "
    f"нужные статьи процитированы в {n_cit_gold}, лишние ссылки в {n_cit_extra}"
)
print(f"вопросов без ответа: {n_no}; отказов: {n_no_ref}")