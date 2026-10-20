from fastapi.testclient import TestClient

from rag_labor_code.chunker import Chunk
from rag_labor_code.service import create_app


class FakeIndex:
    def search_chunks(self, query, k=5):
        text = "Глава 1\nСтатья 152. Оплата сверхурочной работы\nТекст статьи."
        return [Chunk("152#0", "152", "Глава 1", text)][:k]


def test_ask_returns_sources():
    with TestClient(create_app(FakeIndex())) as client:
        r = client.post("/ask", json={"question": "доплата за сверхурочные"})
    assert r.status_code == 200
    body = r.json()
    assert body["sources"][0]["article"] == "152"
    assert "[ст. 152]" in body["answer"]


def test_empty_question_rejected():
    with TestClient(create_app(FakeIndex())) as client:
        r = client.post("/ask", json={"question": ""})
    assert r.status_code == 422


def test_health():
    with TestClient(create_app(FakeIndex())) as client:
        assert client.get("/health").json() == {"status": "ok"}