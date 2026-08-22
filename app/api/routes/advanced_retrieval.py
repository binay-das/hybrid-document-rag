from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.schemas.advanced_retrieval import AdvancedRAGRequest, AdvancedRAGResponse, QueryPreprocessResult
from app.services.advanced_retrieval import (
    MetadataFilter,
    MetadataFilterService,
    ParentDocumentService,
    QueryExpansionService,
    QueryRewritingService,
)
from app.services.generation import GenerationService
from app.services.hybrid import HybridRetrievalService
from app.services.reranker import RerankerService

router = APIRouter(prefix="/advanced", tags=["advanced-retrieval"])


@router.post("/rag", response_model=AdvancedRAGResponse)
def advanced_rag(
    request: AdvancedRAGRequest,
    db: Session = Depends(get_db),
):
    """
    Advanced RAG Pipeline (Phase 11):
    1. Query Rewriting   — clean and optimise the raw query for retrieval
    2. Query Expansion   — inject synonyms / related terms for broader recall
    3. Metadata Filtering — restrict search to specific docs/authors/pages
    4. Hybrid RRF Retrieval — dense + sparse fused
    5. Cross-Encoder Reranking — refine ranking
    6. Parent-Document Retrieval — expand each hit with sibling chunks
    7. Gemini Generation — grounded answer with page-level citations
    """
    if not request.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query string cannot be empty",
        )

    #1. Query Rewriting 
    rewriter = QueryRewritingService()
    rewritten = rewriter.rewrite(request.query) if request.rewrite_query else request.query

    #2. Query Expansion 
    expander = QueryExpansionService()
    if request.expand_query:
        expanded, expansion_terms = expander.expand(rewritten)
    else:
        expanded, expansion_terms = rewritten, []

    #3. Metadata Filtering 
    meta_filter = MetadataFilter()
    allowed_doc_ids: list | None = None
    if request.metadata_filter:
        mf = request.metadata_filter
        meta_filter = MetadataFilter(
            document_ids=mf.document_ids,
            author=mf.author,
            title=mf.title,
            min_page=mf.min_page,
            max_page=mf.max_page,
        )
        filter_svc = MetadataFilterService()
        allowed_doc_ids = filter_svc.resolve_document_ids(db, meta_filter)

        # Short-circuit if metadata filters matched nothing
        if allowed_doc_ids is not None and len(allowed_doc_ids) == 0:
            return AdvancedRAGResponse(
                query_preprocessing=QueryPreprocessResult(
                    original_query=request.query,
                    rewritten_query=rewritten,
                    expanded_query=expanded,
                    expansion_terms=expansion_terms,
                ),
                retrieved_chunks=[],
                answer="I cannot answer this question based on the provided context.",
                citations=[],
                validation={
                    "is_valid": True,
                    "grounding_score": 1.0,
                    "total_citations": 0,
                    "valid_citations": [],
                    "invalid_citations": [],
                    "cited_pages": [],
                },
            )

    #4. Hybrid RRF Retrieval (on expanded query, with metadata doc_id filter) 
    hybrid_service = HybridRetrievalService()

    # If metadata resolved multiple doc_ids, run per-doc and merge
    if allowed_doc_ids and len(allowed_doc_ids) > 1:
        all_candidates: list = []
        per_doc_fetch = max(request.fetch_k // len(allowed_doc_ids), 5)
        for doc_id in allowed_doc_ids:
            candidates = hybrid_service.search_hybrid(
                db=db,
                query=expanded,
                top_k=per_doc_fetch,
                fetch_k=per_doc_fetch,
                rrf_k=request.rrf_k,
                document_id=doc_id,
            )
            all_candidates.extend(candidates)
        # Re-sort merged candidates
        all_candidates.sort(key=lambda x: x["score"], reverse=True)
        hybrid_candidates = all_candidates[: request.fetch_k]
    else:
        single_doc_id = allowed_doc_ids[0] if allowed_doc_ids else None
        hybrid_candidates = hybrid_service.search_hybrid(
            db=db,
            query=expanded,
            top_k=request.fetch_k,
            fetch_k=request.fetch_k,
            rrf_k=request.rrf_k,
            document_id=single_doc_id,
        )

    # Apply page-range filter
    if request.metadata_filter:
        filter_svc = MetadataFilterService()
        hybrid_candidates = filter_svc.apply_to_chunks(hybrid_candidates, meta_filter)

    #5. Cross-Encoder Reranking ---
    reranker = RerankerService()
    # Rerank on original (un-expanded) rewritten query for precision
    reranked = reranker.rerank(
        query=rewritten,
        candidates=hybrid_candidates,
        top_k=request.top_k,
    )

    #6. Parent-Document Retrieval 
    if request.parent_window > 0:
        parent_svc = ParentDocumentService(window_size=request.parent_window)
        final_chunks = parent_svc.fetch_parent_context(db=db, chunks=reranked)
    else:
        final_chunks = reranked

    #7. Gemini Generation & Citation Grounding
    generation_svc = GenerationService()
    gen_result = generation_svc.generate_answer(query=rewritten, chunks=final_chunks)

    return AdvancedRAGResponse(
        query_preprocessing=QueryPreprocessResult(
            original_query=request.query,
            rewritten_query=rewritten,
            expanded_query=expanded,
            expansion_terms=expansion_terms,
        ),
        retrieved_chunks=final_chunks,
        answer=gen_result["answer"],
        citations=gen_result["citations"],
        validation=gen_result["validation"],
    )
