import math
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session

from app.models.document import Chunk


def tokenize(text: str) -> List[str]:
    """Simple alphanumeric tokenizer with lowercasing."""
    if not text:
        return []
    return re.findall(r"\w+", text.lower())


class BM25Service:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

    def search_chunks(
        self,
        db: Session,
        query: str,
        top_k: int = 5,
        document_id: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        query_builder = db.query(Chunk)
        if document_id is not None:
            query_builder = query_builder.filter(Chunk.document_id == document_id)
        chunks = query_builder.all()

        if not chunks:
            return []

        corpus_tokens: List[List[str]] = [tokenize(c.text) for c in chunks]
        N = len(chunks)
        if N == 0:
            return []

        doc_lens = [len(tokens) for tokens in corpus_tokens]
        avgdl = sum(doc_lens) / N if N > 0 else 1.0

        df: Dict[str, int] = defaultdict(int)
        for tokens in corpus_tokens:
            unique_terms = set(tokens)
            for term in unique_terms:
                df[term] += 1

        idf: Dict[str, float] = {}
        for term in set(query_tokens):
            n_q = df.get(term, 0)
            idf[term] = math.log((N - n_q + 0.5) / (n_q + 0.5) + 1.0)

        scored_chunks = []
        for chunk, tokens, d_len in zip(chunks, corpus_tokens, doc_lens):
            if d_len == 0:
                continue

            tf_counts = Counter(tokens)
            score = 0.0
            for q_term in query_tokens:
                if q_term not in tf_counts:
                    continue

                f_q = tf_counts[q_term]
                idf_q = idf.get(q_term, 0.0)

                numerator = f_q * (self.k1 + 1.0)
                denominator = f_q + self.k1 * (1.0 - self.b + self.b * (d_len / avgdl))

                score += idf_q * (numerator / denominator)

            if score > 0:
                scored_chunks.append({
                    "chunk_id": chunk.id,
                    "document_id": chunk.document_id,
                    "page_id": chunk.page_id,
                    "page_number": chunk.page_number,
                    "chunk_index": chunk.chunk_index,
                    "text": chunk.text,
                    "char_count": chunk.char_count,
                    "score": float(score),
                })

        scored_chunks.sort(key=lambda x: x["score"], reverse=True)
        return scored_chunks[:top_k]
