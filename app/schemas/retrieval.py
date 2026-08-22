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

