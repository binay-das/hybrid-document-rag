from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session

from app.services.bm25 import BM25Service
from app.services.embedding import EmbeddingService
from app.services.qdrant import QdrantService


class HybridRetrievalService:
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.qdrant_service = QdrantService()
        self.bm25_service = BM25Service()

    def search_hybrid(
        self,
        db: Session,
        query: str,
        top_k: int = 5,
        fetch_k: int = 20,
        rrf_k: int = 60,
        document_id: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if not query or not query.strip():
            return []

        # 1. Dense retrieval from Qdrant
        query_vector = self.embedding_service.embed_text(query)
        dense_results = self.qdrant_service.search_vectors(
            query_vector=query_vector,
            top_k=fetch_k,
            document_id=document_id,
        )

        # 2. Sparse retrieval via BM25
        sparse_results = self.bm25_service.search_chunks(
            db=db,
            query=query,
            top_k=fetch_k,
            document_id=document_id,
        )

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores: Dict[int, float] = {}
        chunk_map: Dict[int, Dict[str, Any]] = {}

        # Process Dense ranks (1-indexed)
        for rank, item in enumerate(dense_results, start=1):
            c_id = item["chunk_id"]
            chunk_map[c_id] = item
            rrf_scores[c_id] = rrf_scores.get(c_id, 0.0) + (1.0 / (rrf_k + rank))

        # Process Sparse ranks (1-indexed)
        for rank, item in enumerate(sparse_results, start=1):
            c_id = item["chunk_id"]
            if c_id not in chunk_map:
                chunk_map[c_id] = item
            rrf_scores[c_id] = rrf_scores.get(c_id, 0.0) + (1.0 / (rrf_k + rank))

        # 4. Format & Sort fused results
        fused_results = []
        for c_id, score in rrf_scores.items():
            item_copy = dict(chunk_map[c_id])
            item_copy["score"] = float(score)
            fused_results.append(item_copy)

        fused_results.sort(key=lambda x: x["score"], reverse=True)
        return fused_results[:top_k]
