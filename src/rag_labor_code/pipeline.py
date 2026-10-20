import hashlib
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from rag_labor_code.chunker import chunk_articles
from rag_labor_code.docx_to_text import docx_to_text
from rag_labor_code.parser import Article, parse

SOURCE = Path("data/raw/tk_rf.docx")
TEXT = Path("data/raw/tk_rf.txt")
STATE = Path("data/pipeline_state.json")
INDEX_DIR = Path("data/index")
MIN_MRR = 0.70  # проверка на «не сломано», а не обещание качества


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def source_changed(source: Path, state_path: Path = STATE) -> bool:
    if not state_path.exists():
        return True
    state = json.loads(state_path.read_text(encoding="utf-8"))
    return state.get("source_sha256") != file_sha256(source)


def mark_processed(source: Path, state_path: Path = STATE, **extra) -> None:
    state = {"source_sha256": file_sha256(source), **extra}
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")


def validate_articles(arts: list[Article]) -> None:
    numbers = [a.number for a in arts]
    if not 400 <= len(arts) <= 700:
        raise ValueError(f"неожиданное число статей: {len(arts)}")
    if numbers[0] != "1":
        raise ValueError(f"первая статья {numbers[0]}, ожидалась 1")
    if len(numbers) != len(set(numbers)):
        raise ValueError("есть повторяющиеся номера статей")
    if any(not a.paragraphs for a in arts):
        raise ValueError("есть статьи без текста")


def load_articles(text: Path = TEXT) -> list[Article]:
    return parse(text.read_text(encoding="utf-8"))


def prepare() -> list[Article]:
    docx_to_text(SOURCE, TEXT)
    arts = load_articles()
    validate_articles(arts)
    return arts


def build_and_evaluate(arts: list[Article]):
    # тяжёлые импорты здесь, чтобы тесты и проверка хеша не грузили torch
    from rag_labor_code.dense import DenseIndex
    from rag_labor_code.evaluate_retrieval import evaluate, load_gold

    chunks = [replace(c, text=f"{c.chapter}\n{c.text}") for c in chunk_articles(arts, 1500)]
    index = DenseIndex(chunks)
    rows = load_gold("dev")
    recall, mrr = evaluate(index, rows)
    metrics = {
        "mrr": round(mrr, 3),
        "recall@5": round(recall[5], 3),
        "n_dev_questions": len(rows),
    }
    return index, metrics


def quality_gate(metrics: dict) -> None:
    if metrics["mrr"] < MIN_MRR:
        raise ValueError(f"MRR {metrics['mrr']} ниже порога {MIN_MRR}, индекс не публикуем")


def publish(index, metrics: dict, version: str) -> None:
    out = INDEX_DIR / version
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "emb.npy", index.emb)
    chunks_json = json.dumps([asdict(c) for c in index.chunks], ensure_ascii=False)
    (out / "chunks.json").write_text(chunks_json, encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")
    (INDEX_DIR / "CURRENT.txt").write_text(version, encoding="utf-8")  # указатель меняем последним


def build_and_publish(arts: list[Article]) -> str:
    index, metrics = build_and_evaluate(arts)
    quality_gate(metrics)
    version = file_sha256(SOURCE)[:8]
    publish(index, metrics, version)
    mark_processed(SOURCE, version=version, **metrics)  # состояние обновляем только после успеха
    return f"опубликована версия {version}: {metrics}"


def run(force: bool = False) -> str:
    if not force and not source_changed(SOURCE):
        return "источник не изменился, пропускаем"
    return build_and_publish(prepare())


if __name__ == "__main__":
    print(run(force="--force" in sys.argv))