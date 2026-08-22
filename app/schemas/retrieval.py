from typing import List, Optional
from pydantic import BaseModel, Field


class DenseSearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=100)
    document_id: Optional[int] = None


class SearchResultItem(BaseModel):
    chunk_id: int
    document_id: int
    page_id: Optional[int] = None
    page_number: int
    chunk_index: int
    text: str
    char_count: int
    score: float
    cross_encoder_score: Optional[float] = None



class DenseSearchResponse(BaseModel):
    query: str
    top_k: int
    results: List[SearchResultItem]


class DenseTestCaseSchema(BaseModel):
    query: str
    relevant_chunk_ids: List[int]
    document_id: Optional[int] = None


class DenseEvaluationRequest(BaseModel):
    top_k: int = Field(default=5, ge=1, le=100)
    test_cases: List[DenseTestCaseSchema]


class QueryEvalResultSchema(BaseModel):
    query: str
    relevant_chunk_ids: List[int]
    retrieved_chunk_ids: List[int]
    recall_at_k: float
    hit_at_k: float
    top_chunks: List[SearchResultItem]


class DenseEvaluationResponse(BaseModel):
    total_queries: int
    top_k: int
    mean_recall_at_k: float
    hit_rate_at_k: float
    query_results: List[QueryEvalResultSchema]


class SparseSearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=100)
    document_id: Optional[int] = None
    k1: float = Field(default=1.5, ge=0.0)
    b: float = Field(default=0.75, ge=0.0, le=1.0)


class SparseSearchResponse(BaseModel):
    query: str
    top_k: int
    results: List[SearchResultItem]


class SparseTestCaseSchema(BaseModel):
    query: str
    relevant_chunk_ids: List[int]
    document_id: Optional[int] = None


class SparseEvaluationRequest(BaseModel):
    top_k: int = Field(default=5, ge=1, le=100)
    k1: float = Field(default=1.5, ge=0.0)
    b: float = Field(default=0.75, ge=0.0, le=1.0)
    test_cases: List[SparseTestCaseSchema]


class SparseEvaluationResponse(BaseModel):
    total_queries: int
    top_k: int
    mean_recall_at_k: float
    hit_rate_at_k: float
    query_results: List[QueryEvalResultSchema]


class HybridSearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=100)
    fetch_k: int = Field(default=20, ge=1, le=200)
    rrf_k: int = Field(default=60, ge=1)
    document_id: Optional[int] = None


class HybridSearchResponse(BaseModel):
    query: str
    top_k: int
    rrf_k: int
    results: List[SearchResultItem]


class ComparisonQueryResultSchema(BaseModel):
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


class ComparativeEvaluationRequest(BaseModel):
    top_k: int = Field(default=5, ge=1, le=100)
    fetch_k: int = Field(default=20, ge=1, le=200)
    rrf_k: int = Field(default=60, ge=1)
    test_cases: List[DenseTestCaseSchema]


class ComparativeEvaluationResponse(BaseModel):
    total_queries: int
    top_k: int
    dense_mean_recall: float
    dense_hit_rate: float
    sparse_mean_recall: float
    sparse_hit_rate: float
    hybrid_mean_recall: float
    hybrid_hit_rate: float
    query_comparisons: List[ComparisonQueryResultSchema]


class RerankSearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=100)
    fetch_k: int = Field(default=20, ge=1, le=200)
    rrf_k: int = Field(default=60, ge=1)
    document_id: Optional[int] = None


class RerankSearchResponse(BaseModel):
    query: str
    top_k: int
    results: List[SearchResultItem]


class RerankQueryComparisonSchema(BaseModel):
    query: str
    relevant_chunk_ids: List[int]
    hybrid_retrieved_ids: List[int]
    hybrid_mrr: float
    hybrid_ndcg: float
    reranked_retrieved_ids: List[int]
    reranked_mrr: float
    reranked_ndcg: float
    reranked_top_chunks: List[SearchResultItem]


class RerankEvaluationRequest(BaseModel):
    top_k: int = Field(default=5, ge=1, le=100)
    fetch_k: int = Field(default=20, ge=1, le=200)
    rrf_k: int = Field(default=60, ge=1)
    test_cases: List[DenseTestCaseSchema]


class RerankEvaluationResponse(BaseModel):
    total_queries: int
    top_k: int
    hybrid_mean_recall: float
    hybrid_hit_rate: float
    hybrid_mrr: float
    hybrid_ndcg: float
    reranked_mean_recall: float
    reranked_hit_rate: float
    reranked_mrr: float
    reranked_ndcg: float
    query_comparisons: List[RerankQueryComparisonSchema]



