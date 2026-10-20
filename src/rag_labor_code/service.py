from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field

from rag_labor_code.chunker import Chunk, chunk_articles
from rag_labor_code.parser import parse

NOTE = "Учебный проект, не юридическая консультация. Ответ без языковой модели: показан наиболее релевантный фрагмент кодекса."


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    k: int = Field(default=5, ge=1, le=10)


class Source(BaseModel):
    article: str
    text: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    note: str


def build_index():
    # импорт здесь, чтобы тесты не загружали torch и модель
    from rag_labor_code.dense import DenseIndex

    arts = parse(Path("data/raw/tk_rf.txt").read_text(encoding="utf-8"))
    chunks = [replace(c, text=f"{c.chapter}\n{c.text}") for c in chunk_articles(arts, 1500)]
    return DenseIndex(chunks)


def extractive_answer(chunks: list[Chunk]) -> str:
    best = chunks[0]
    body = best.text.split("\n", 1)[-1]  # первая строка чанка это название главы
    return f"{body[:700]} [ст. {best.article}]"


def create_app(index=None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.index = index if index is not None else build_index()
        yield

    app = FastAPI(title="Labor Code RAG", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/ask", response_model=AskResponse)
    def ask(req: AskRequest):
        chunks = app.state.index.search_chunks(req.question, req.k)
        return AskResponse(
            answer=extractive_answer(chunks),
            sources=[Source(article=c.article, text=c.text) for c in chunks],
            note=NOTE,
        )

    return app