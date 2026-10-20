import pytest

from rag_labor_code.parser import Article
from rag_labor_code.pipeline import mark_processed, source_changed, validate_articles


def make(n):
    return [Article("", "", "", str(i), "t", ["x"]) for i in range(1, n + 1)]


def test_source_changed_lifecycle(tmp_path):
    src = tmp_path / "src.docx"
    src.write_bytes(b"v1")
    state = tmp_path / "state.json"
    assert source_changed(src, state)
    mark_processed(src, state)
    assert not source_changed(src, state)
    src.write_bytes(b"v2")
    assert source_changed(src, state)


def test_validate_ok():
    validate_articles(make(500))


def test_validate_too_few():
    with pytest.raises(ValueError):
        validate_articles(make(10))


def test_validate_duplicates():
    arts = make(500)
    arts[-1].number = "5"
    with pytest.raises(ValueError):
        validate_articles(arts)