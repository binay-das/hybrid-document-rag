from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from app.services.bm25 import BM25Service
from app.services.embedding import EmbeddingService
from app.services.qdrant import QdrantService


@dataclass
class DenseTestCase:
    query: str
    relevant_chunk_ids: List[int]
    document_id: Optional[int] = None


@dataclass
class SparseTestCase:
    query: str
    relevant_chunk_ids: List[int]
    document_id: Optional[int] = None


@dataclass
class QueryEvalResult:
    query: str
    relevant_chunk_ids: List[int]
    retrieved_chunk_ids: List[int]
    recall_at_k: float
    hit_at_k: float
    top_chunks: List[Dict[str, Any]]


@dataclass
class EvaluationReport:
    total_queries: int
    top_k: int
    mean_recall_at_k: float
    hit_rate_at_k: float
    query_results: List[QueryEvalResult]


class EvaluationService:
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.qdrant_service = QdrantService()

    def evaluate_dense_retrieval(
        self,
        test_cases: List[DenseTestCase],
        top_k: int = 5,
    ) -> EvaluationReport:
        if not test_cases:
            return EvaluationReport(
                total_queries=0,
                top_k=top_k,
                mean_recall_at_k=0.0,
                hit_rate_at_k=0.0,
                query_results=[],
            )

        eval_results: List[QueryEvalResult] = []
        recalls: List[float] = []
        hits: List[float] = []

        for tc in test_cases:
            query_vector = self.embedding_service.embed_text(tc.query)
            search_results = self.qdrant_service.search_vectors(
                query_vector=query_vector,
                top_k=top_k,
                document_id=tc.document_id,
            )

            retrieved_chunk_ids = [res["chunk_id"] for res in search_results]
            gt_set = set(tc.relevant_chunk_ids)

            intersection = set(retrieved_chunk_ids).intersection(gt_set)
            recall = len(intersection) / len(gt_set) if gt_set else 0.0
            hit = 1.0 if len(intersection) > 0 else 0.0

            recalls.append(recall)
            hits.append(hit)

            eval_results.append(
                QueryEvalResult(
                    query=tc.query,
                    relevant_chunk_ids=tc.relevant_chunk_ids,
                    retrieved_chunk_ids=retrieved_chunk_ids,
                    recall_at_k=round(recall, 4),
                    hit_at_k=hit,
                    top_chunks=search_results,
                )
            )

        total = len(test_cases)
        mean_recall = sum(recalls) / total
        hit_rate = sum(hits) / total

        return EvaluationReport(
            total_queries=total,
            top_k=top_k,
            mean_recall_at_k=round(mean_recall, 4),
            hit_rate_at_k=round(hit_rate, 4),
            query_results=eval_results,
        )

    def evaluate_sparse_retrieval(
        self,
        db: Session,
        test_cases: List[SparseTestCase],
        top_k: int = 5,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> EvaluationReport:
        if not test_cases:
            return EvaluationReport(
                total_queries=0,
                top_k=top_k,
                mean_recall_at_k=0.0,
                hit_rate_at_k=0.0,
                query_results=[],
            )

        bm25_service = BM25Service(k1=k1, b=b)
        eval_results: List[QueryEvalResult] = []
        recalls: List[float] = []
        hits: List[float] = []

        for tc in test_cases:
            search_results = bm25_service.search_chunks(
                db=db,
                query=tc.query,
                top_k=top_k,
                document_id=tc.document_id,
            )

            retrieved_chunk_ids = [res["chunk_id"] for res in search_results]
            gt_set = set(tc.relevant_chunk_ids)

            intersection = set(retrieved_chunk_ids).intersection(gt_set)
            recall = len(intersection) / len(gt_set) if gt_set else 0.0
            hit = 1.0 if len(intersection) > 0 else 0.0

            recalls.append(recall)
            hits.append(hit)

            eval_results.append(
                QueryEvalResult(
                    query=tc.query,
                    relevant_chunk_ids=tc.relevant_chunk_ids,
                    retrieved_chunk_ids=retrieved_chunk_ids,
                    recall_at_k=round(recall, 4),
                    hit_at_k=hit,
                    top_chunks=search_results,
                )
            )

        total = len(test_cases)
        mean_recall = sum(recalls) / total
        hit_rate = sum(hits) / total

        return EvaluationReport(
            total_queries=total,
            top_k=top_k,
            mean_recall_at_k=round(mean_recall, 4),
            hit_rate_at_k=round(hit_rate, 4),
            query_results=eval_results,
        )

