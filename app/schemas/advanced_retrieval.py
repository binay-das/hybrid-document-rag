from typing import List, Optional
from pydantic import BaseModel, Field
from app.schemas.retrieval import SearchResultItem


# Shared
class MetadataFilterSchema(BaseModel):
    document_ids: Optional[List[int]] = None
    author: Optional[str] = None
    title: Optional[str] = None
    min_page: Optional[int] = None
    max_page: Optional[int] = None


# Advanced RAG Request / Response
class AdvancedRAGRequest(BaseModel):
    query: str

    # Retrieval knobs
    top_k: int = Field(default=5, ge=1, le=100)
    fetch_k: int = Field(default=20, ge=1, le=200)
    rrf_k: int = Field(default=60, ge=1)

    # Feature toggles
    rewrite_query: bool = True
    expand_query: bool = True
    parent_window: int = Field(default=1, ge=0, le=3, description="Sibling chunks on each side to include as parent context")
    metadata_filter: Optional[MetadataFilterSchema] = None


class QueryPreprocessResult(BaseModel):
    original_query: str
    rewritten_query: str
    expanded_query: str
    expansion_terms: List[str]


class AdvancedRAGResponse(BaseModel):
    query_preprocessing: QueryPreprocessResult
    retrieved_chunks: List[SearchResultItem]
    answer: str
    citations: list
    validation: dict
