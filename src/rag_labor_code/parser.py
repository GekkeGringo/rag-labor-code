import re
from dataclasses import dataclass, field
from pathlib import Path

PART_RE = re.compile(r"^ЧАСТЬ\s+\S+")
SECTION_RE = re.compile(r"^Раздел\s+[IVXL]+(?:\.\d+)?\.")
CHAPTER_RE = re.compile(r"^Глава\s+\d+(?:\.\d+)?\.")
ARTICLE_RE = re.compile(r"^Статья\s+(\d+(?:\.\d+)*(?:-\d+)?)\.\s*(.*)$")
# пометка об изменениях в конце строки: "(в ред. ... N 185-ФЗ)"
NOTE_RE = re.compile(r"\s*\([^()]*\d+-ФЗ[^()]*\)\s*$")
REPEALED_RE = re.compile(r"^[-–—\s]*Утратила силу\.?$")
EDITORIAL_RE = re.compile(
    r"^Действие .*\(в редакции Федерального закона .*\) распространяется на правоотношения"
)


@dataclass
class Article:
    part: str
    section: str
    chapter: str
    number: str  # строкой, потому что бывают номера вроде "5.1" и "348.11-1"
    title: str
    paragraphs: list[str] = field(default_factory=list)


def clean(line: str) -> str:
    line = line.strip()
    while (m := NOTE_RE.search(line)):
        line = line[: m.start()].rstrip()
    return line


def parse(text: str) -> list[Article]:
    articles: list[Article] = []
    part = section = chapter = ""
    current: Article | None = None
    started = False
    for raw in text.splitlines():
        line = clean(raw)
        if not line:
            continue
        if EDITORIAL_RE.match(line):
            continue
        if PART_RE.match(line):
            part, started = line, True
            continue
        if not started:
            continue
        if SECTION_RE.match(line):
            section = line
        elif CHAPTER_RE.match(line):
            chapter = line
        elif (m := ARTICLE_RE.match(line)):
            current = Article(part, section, chapter, m.group(1), m.group(2))
            articles.append(current)
        elif current is not None:
            current.paragraphs.append(line)

    # подпись под кодексом приклеивается к последней статье: отрезаем её
    if articles and "Президент" in articles[-1].paragraphs:
        last = articles[-1].paragraphs
        del last[last.index("Президент"):]

    return [a for a in articles if not (REPEALED_RE.match(a.title) and not a.paragraphs)]


if __name__ == "__main__":
    text = Path("data/raw/tk_rf.txt").read_text(encoding="utf-8")
    arts = parse(text)
    numbers = [a.number for a in arts]
    print("статей:", len(arts))
    print("первая:", numbers[0], "| последняя:", numbers[-1])
    print("повторяющихся номеров:", len(numbers) - len(set(numbers)))
    print("статей без текста:", sum(1 for a in arts if not a.paragraphs))
    print("последняя строка статьи 424:", arts[-1].paragraphs[-1][:80])