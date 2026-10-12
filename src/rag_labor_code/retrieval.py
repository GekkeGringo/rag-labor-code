import re

from nltk.stem.snowball import SnowballStemmer
from rank_bm25 import BM25Okapi

from rag_labor_code.chunker import Chunk

_stem = SnowballStemmer("russian")
_WORD = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return [_stem.stem(w) for w in _WORD.findall(text.lower())]


class BM25Index:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(c.text) for c in chunks])

    def search_articles(self, query: str, k: int = 10) -> list[str]:
        scores = self.bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        seen: set[str] = set()
        result: list[str] = []
        for i in order:
            art = self.chunks[i].article
            if art not in seen:
                seen.add(art)
                result.append(art)
                if len(result) == k:
                    break
        return result