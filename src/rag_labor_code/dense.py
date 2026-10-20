import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from rag_labor_code.chunker import Chunk

MODEL = "intfloat/multilingual-e5-small"


class DenseIndex:
    def __init__(self, chunks: list[Chunk], model_name: str = MODEL, emb=None):
        self.chunks = chunks
        self.model = SentenceTransformer(model_name)
        if emb is None:
            texts = ["passage: " + c.text for c in chunks]
            emb = self.model.encode(
                texts, batch_size=32, normalize_embeddings=True, show_progress_bar=True
            )
        self.emb = emb

    @classmethod
    def load(cls, index_dir, model_name: str = MODEL) -> "DenseIndex":
        index_dir = Path(index_dir)
        raw = json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))
        chunks = [Chunk(**c) for c in raw]
        return cls(chunks, model_name, emb=np.load(index_dir / "emb.npy"))

    def search_articles(self, query: str, k: int = 10) -> list[str]:
        q = self.model.encode(["query: " + query], normalize_embeddings=True)[0]
        scores = self.emb @ q
        seen: set[str] = set()
        result: list[str] = []
        for i in np.argsort(-scores):
            art = self.chunks[i].article
            if art not in seen:
                seen.add(art)
                result.append(art)
                if len(result) == k:
                    break
        return result

    def search_chunks(self, query: str, k: int = 5) -> list[Chunk]:
        q = self.model.encode(["query: " + query], normalize_embeddings=True)[0]
        order = np.argsort(-(self.emb @ q))[:k]
        return [self.chunks[i] for i in order]