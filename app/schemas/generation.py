from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from app.schemas.retrieval import SearchResultItem


class PageCitationSchema(BaseModel):
    document_id: int
    page_number: int


class CitationValidationSchema(BaseModel):
    is_valid: bool
    grounding_score: float
    total_citations: int
    valid_citations: List[PageCitationSchema]
    invalid_citations: List[PageCitationSchema]
    cited_pages: List[PageCitationSchema]


class RAGGenerateRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=100)
    fetch_k: int = Field(default=20, ge=1, le=200)
    rrf_k: int = Field(default=60, ge=1)
    document_id: Optional[int] = None


class RAGGenerateResponse(BaseModel):
    query: str
    answer: str
    citations: List[PageCitationSchema]
    validation: CitationValidationSchema
    retrieved_chunks: List[SearchResultItem]
