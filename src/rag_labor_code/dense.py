import numpy as np
from sentence_transformers import SentenceTransformer

from rag_labor_code.chunker import Chunk

MODEL = "intfloat/multilingual-e5-small"


class DenseIndex:
    def __init__(self, chunks: list[Chunk], model_name: str = MODEL):
        self.chunks = chunks
        self.model = SentenceTransformer(model_name)
        texts = ["passage: " + c.text for c in chunks]
        self.emb = self.model.encode(
            texts, batch_size=32, normalize_embeddings=True, show_progress_bar=True
        )

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