from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.schemas.generation import RAGGenerateRequest, RAGGenerateResponse
from app.services.generation import GenerationService
from app.services.hybrid import HybridRetrievalService
from app.services.reranker import RerankerService

router = APIRouter(prefix="/generate", tags=["generation"])


@router.post("/rag", response_model=RAGGenerateResponse)
def generate_rag_answer(
    request: RAGGenerateRequest,
    db: Session = Depends(get_db),
):
    """
    RAG Generation Endpoint:
    1. Retrieves candidate chunks using Hybrid RRF Search.
    2. Reranks candidates using Cross-Encoder Reranker.
    3. Builds grounded context and generates an answer using Gemini.
    4. Extracts page-level citations and validates grounding.
    """
    if not request.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query string cannot be empty",
        )

    # 1. Hybrid RRF Retrieval
    hybrid_service = HybridRetrievalService()
    hybrid_candidates = hybrid_service.search_hybrid(
        db=db,
        query=request.query,
        top_k=request.fetch_k,
        fetch_k=request.fetch_k,
        rrf_k=request.rrf_k,
        document_id=request.document_id,
    )

    # 2. Cross-Encoder Reranking
    reranker = RerankerService()
    reranked_chunks = reranker.rerank(
        query=request.query,
        candidates=hybrid_candidates,
        top_k=request.top_k,
    )

    # 3. Generation & Citation Grounding Validation
    generation_service = GenerationService()
    gen_result = generation_service.generate_answer(
        query=request.query,
        chunks=reranked_chunks,
    )

    return RAGGenerateResponse(
        query=request.query,
        answer=gen_result["answer"],
        citations=gen_result["citations"],
        validation=gen_result["validation"],
        retrieved_chunks=reranked_chunks,
    )
