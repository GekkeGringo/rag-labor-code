import numpy as np
import pytest

from rag_labor_code import pipeline
from rag_labor_code.chunker import Chunk


class FakeIndex:
    def __init__(self):
        self.emb = np.zeros((1, 3), dtype="float32")
        self.chunks = [Chunk("1#0", "1", "Глава 1", "Глава 1\nСтатья 1. Тест\nТекст.")]


def test_publish_writes_version_and_pointer(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "INDEX_DIR", tmp_path)
    pipeline.publish(FakeIndex(), {"mrr": 0.8}, "abc12345")
    assert (tmp_path / "abc12345" / "emb.npy").exists()
    assert (tmp_path / "abc12345" / "chunks.json").exists()
    assert (tmp_path / "CURRENT.txt").read_text(encoding="utf-8") == "abc12345"


def test_quality_gate_blocks_low_mrr():
    with pytest.raises(ValueError):
        pipeline.quality_gate({"mrr": 0.5})