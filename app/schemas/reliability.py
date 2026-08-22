from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from app.schemas.retrieval import SearchResultItem
from app.schemas.advanced_retrieval import MetadataFilterSchema, QueryPreprocessResult


class GuardrailStatusSchema(BaseModel):
    is_rejected: bool
    is_adversarial: bool
    rejection_reason: Optional[str] = None


class ConfidenceScoreSchema(BaseModel):
    retrieval_confidence: float
    grounding_confidence: float
    overall_confidence: float
    is_low_confidence: bool
    threshold: float


class WebSourceItemSchema(BaseModel):
    title: str
    url: str
    citation_tag: str


class WebFallbackResultSchema(BaseModel):
    triggered: bool
    query: str
    results_count: int
    web_sources: List[WebSourceItemSchema]


class ReliableRAGRequest(BaseModel):
    query: str

    # Retrieval configuration
    top_k: int = Field(default=5, ge=1, le=100)
    fetch_k: int = Field(default=20, ge=1, le=200)
    rrf_k: int = Field(default=60, ge=1)

    # Reliability controls
    confidence_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    enable_guardrails: bool = True
    enable_web_fallback: bool = True

    # Advanced retrieval features
    rewrite_query: bool = True
    expand_query: bool = True
    parent_window: int = Field(default=1, ge=0, le=3)
    metadata_filter: Optional[MetadataFilterSchema] = None


class ReliableRAGResponse(BaseModel):
    query: str
    guardrail: GuardrailStatusSchema
    confidence: ConfidenceScoreSchema
    used_web_fallback: bool
    web_search_details: Optional[WebFallbackResultSchema] = None
    query_preprocessing: Optional[QueryPreprocessResult] = None
    retrieved_chunks: List[SearchResultItem]
    answer: str
    citations: list
    validation: dict
