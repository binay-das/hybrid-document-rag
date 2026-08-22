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


from app.services.hybrid import HybridRetrievalService


@dataclass
class ComparisonQueryResult:
    query: str
    relevant_chunk_ids: List[int]
    dense_retrieved_ids: List[int]
    dense_recall: float
    dense_hit: float
    sparse_retrieved_ids: List[int]
    sparse_recall: float
    sparse_hit: float
    hybrid_retrieved_ids: List[int]
    hybrid_recall: float
    hybrid_hit: float


@dataclass
class ComparativeEvaluationReport:
    total_queries: int
    top_k: int
    dense_mean_recall: float
    dense_hit_rate: float
    sparse_mean_recall: float
    sparse_hit_rate: float
    hybrid_mean_recall: float
    hybrid_hit_rate: float
    query_comparisons: List[ComparisonQueryResult]


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

    def evaluate_comparative_retrieval(
        self,
        db: Session,
        test_cases: List[DenseTestCase],
        top_k: int = 5,
        fetch_k: int = 20,
        rrf_k: int = 60,
    ) -> ComparativeEvaluationReport:
        if not test_cases:
            return ComparativeEvaluationReport(
                total_queries=0,
                top_k=top_k,
                dense_mean_recall=0.0,
                dense_hit_rate=0.0,
                sparse_mean_recall=0.0,
                sparse_hit_rate=0.0,
                hybrid_mean_recall=0.0,
                hybrid_hit_rate=0.0,
                query_comparisons=[],
            )

        bm25_service = BM25Service()
        hybrid_service = HybridRetrievalService()

        comparisons: List[ComparisonQueryResult] = []
        dense_recalls, dense_hits = [], []
        sparse_recalls, sparse_hits = [], []
        hybrid_recalls, hybrid_hits = [], []

        for tc in test_cases:
            gt_set = set(tc.relevant_chunk_ids)

            # 1. Dense Search
            q_vec = self.embedding_service.embed_text(tc.query)
            d_res = self.qdrant_service.search_vectors(query_vector=q_vec, top_k=top_k, document_id=tc.document_id)
            d_ids = [r["chunk_id"] for r in d_res]
            d_inter = set(d_ids).intersection(gt_set)
            d_recall = len(d_inter) / len(gt_set) if gt_set else 0.0
            d_hit = 1.0 if len(d_inter) > 0 else 0.0
            dense_recalls.append(d_recall)
            dense_hits.append(d_hit)

            # 2. Sparse Search
            s_res = bm25_service.search_chunks(db=db, query=tc.query, top_k=top_k, document_id=tc.document_id)
            s_ids = [r["chunk_id"] for r in s_res]
            s_inter = set(s_ids).intersection(gt_set)
            s_recall = len(s_inter) / len(gt_set) if gt_set else 0.0
            s_hit = 1.0 if len(s_inter) > 0 else 0.0
            sparse_recalls.append(s_recall)
            sparse_hits.append(s_hit)

            # 3. Hybrid Search
            h_res = hybrid_service.search_hybrid(
                db=db,
                query=tc.query,
                top_k=top_k,
                fetch_k=fetch_k,
                rrf_k=rrf_k,
                document_id=tc.document_id,
            )
            h_ids = [r["chunk_id"] for r in h_res]
            h_inter = set(h_ids).intersection(gt_set)
            h_recall = len(h_inter) / len(gt_set) if gt_set else 0.0
            h_hit = 1.0 if len(h_inter) > 0 else 0.0
            hybrid_recalls.append(h_recall)
            hybrid_hits.append(h_hit)

            comparisons.append(
                ComparisonQueryResult(
                    query=tc.query,
                    relevant_chunk_ids=tc.relevant_chunk_ids,
                    dense_retrieved_ids=d_ids,
                    dense_recall=round(d_recall, 4),
                    dense_hit=d_hit,
                    sparse_retrieved_ids=s_ids,
                    sparse_recall=round(s_recall, 4),
                    sparse_hit=s_hit,
                    hybrid_retrieved_ids=h_ids,
                    hybrid_recall=round(h_recall, 4),
                    hybrid_hit=h_hit,
                )
            )

        total = len(test_cases)
        return ComparativeEvaluationReport(
            total_queries=total,
            top_k=top_k,
            dense_mean_recall=round(sum(dense_recalls) / total, 4),
            dense_hit_rate=round(sum(dense_hits) / total, 4),
            sparse_mean_recall=round(sum(sparse_recalls) / total, 4),
            sparse_hit_rate=round(sum(sparse_hits) / total, 4),
            hybrid_mean_recall=round(sum(hybrid_recalls) / total, 4),
            hybrid_hit_rate=round(sum(hybrid_hits) / total, 4),
            query_comparisons=comparisons,
        )


