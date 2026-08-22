from fastapi import APIRouter, HTTPException, status

from app.schemas.retrieval import (
    DenseEvaluationRequest,
    DenseEvaluationResponse,
    DenseSearchRequest,
    DenseSearchResponse,
)
from app.services.embedding import EmbeddingService
from app.services.evaluation import DenseTestCase, EvaluationService
from app.services.qdrant import QdrantService

router = APIRouter(prefix="/search", tags=["retrieval"])


@router.post("/dense", response_model=DenseSearchResponse)
def dense_vector_search(request: DenseSearchRequest):
    if not request.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query string cannot be empty",
        )

    embedding_service = EmbeddingService()
    query_vector = embedding_service.embed_text(request.query)

    qdrant_service = QdrantService()
    results = qdrant_service.search_vectors(
        query_vector=query_vector,
        top_k=request.top_k,
        document_id=request.document_id,
    )

    return DenseSearchResponse(
        query=request.query,
        top_k=request.top_k,
        results=results,
    )


@router.post("/dense/evaluate", response_model=DenseEvaluationResponse)
def evaluate_dense_retrieval(request: DenseEvaluationRequest):
    if not request.test_cases:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="test_cases list cannot be empty",
        )

    test_cases = [
        DenseTestCase(
            query=tc.query,
            relevant_chunk_ids=tc.relevant_chunk_ids,
            document_id=tc.document_id,
        )
        for tc in request.test_cases
    ]

    eval_service = EvaluationService()
    report = eval_service.evaluate_dense_retrieval(
        test_cases=test_cases,
        top_k=request.top_k,
    )

    return report
