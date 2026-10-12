from rag_labor_code.chunker import chunk_articles
from rag_labor_code.parser import Article


def make_article():
    return Article("", "", "Глава 1", "5", "Заголовок", ["а" * 400, "б" * 400, "в" * 400])


def test_whole_article_is_one_chunk():
    assert len(chunk_articles([make_article()])) == 1


def test_long_article_is_split_with_header():
    chunks = chunk_articles([make_article()], max_chars=1000)
    assert len(chunks) == 2
    assert all(c.text.startswith("Статья 5. Заголовок") for c in chunks)