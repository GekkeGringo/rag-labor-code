import statistics
from dataclasses import dataclass
from pathlib import Path

from rag_labor_code.parser import Article, parse


@dataclass
class Chunk:
    id: str
    article: str
    chapter: str
    text: str


def chunk_articles(articles: list[Article], max_chars: int | None = None) -> list[Chunk]:
    """max_chars=None: одна статья = один чанк. Иначе абзацы упаковываются до max_chars."""
    chunks: list[Chunk] = []
    for a in articles:
        header = f"Статья {a.number}. {a.title}"
        if max_chars is None:
            groups = [a.paragraphs]
        else:
            groups, cur, size = [], [], 0
            for p in a.paragraphs:
                if cur and size + len(p) > max_chars:
                    groups.append(cur)
                    cur, size = [], 0
                cur.append(p)
                size += len(p) + 1
            if cur:
                groups.append(cur)
        for i, g in enumerate(groups or [[]]):
            chunks.append(Chunk(f"{a.number}#{i}", a.number, a.chapter, "\n".join([header, *g])))
    return chunks


if __name__ == "__main__":
    arts = parse(Path("data/raw/tk_rf.txt").read_text(encoding="utf-8"))
    for name, limit in [("статья целиком", None), ("до 1500 симв.", 1500), ("до 800 симв.", 800)]:
        lens = [len(c.text) for c in chunk_articles(arts, limit)]
        print(
            f"{name:16} чанков: {len(lens):5} | медиана: {int(statistics.median(lens)):5}"
            f" | макс: {max(lens):6} | длиннее 2000: {sum(n > 2000 for n in lens)}"
        )