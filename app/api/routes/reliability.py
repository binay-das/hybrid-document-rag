from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.schemas.advanced_retrieval import QueryPreprocessResult
from app.schemas.reliability import (
    ConfidenceScoreSchema,
    GuardrailStatusSchema,
    ReliableRAGRequest,
    ReliableRAGResponse,
    WebFallbackResultSchema,
    WebSourceItemSchema,
)
from app.services.advanced_retrieval import (
    MetadataFilter,
    MetadataFilterService,
    ParentDocumentService,
    QueryExpansionService,
    QueryRewritingService,
)
from app.services.generation import GenerationService
from app.services.hybrid import HybridRetrievalService
from app.services.reliability import (
    ConfidenceScorer,
    QueryGuardrailService,
    WebSearchFallbackService,
)
from app.services.reranker import RerankerService

router = APIRouter(prefix="/reliability", tags=["reliability"])


@router.post("/rag", response_model=ReliableRAGResponse)
def reliable_rag(
    request: ReliableRAGRequest,
    db: Session = Depends(get_db),
):
    """
    Phase 12: Reliable RAG Pipeline
    1. Guardrails — reject adversarial / prompt-injection / empty queries
    2. Advanced Retrieval — query rewriting, expansion, metadata filtering, hybrid search, reranking, parent docs
    3. Grounded Generation — generate answer & compute citations
    4. Confidence Scoring — measure retrieval & grounding quality
    5. Web Search Fallback — perform external web search when context confidence is low
    """
    # 1. Guardrail check
    guardrail_svc = QueryGuardrailService()
    guardrail_res = guardrail_svc.check_query(request.query)

    if guardrail_res["is_rejected"]:
        return ReliableRAGResponse(
            query=request.query,
            guardrail=GuardrailStatusSchema(**guardrail_res),
            confidence=ConfidenceScoreSchema(
                retrieval_confidence=0.0,
                grounding_confidence=0.0,
                overall_confidence=0.0,
                is_low_confidence=True,
                threshold=request.confidence_threshold,
            ),
            used_web_fallback=False,
            web_search_details=None,
            query_preprocessing=QueryPreprocessResult(
                original_query=request.query,
                rewritten_query=request.query,
                expanded_query=request.query,
                expansion_terms=[],
            ),
            retrieved_chunks=[],
            answer=f"Query rejected: {guardrail_res['rejection_reason']}",
            citations=[],
            validation={
                "is_valid": False,
                "grounding_score": 0.0,
                "total_citations": 0,
                "valid_citations": [],
                "invalid_citations": [],
                "cited_pages": [],
            },
        )

    # 2. Query Rewriting & Expansion
    rewriter = QueryRewritingService()
    rewritten = rewriter.rewrite(request.query) if request.rewrite_query else request.query

    expander = QueryExpansionService()
    if request.expand_query:
        expanded, expansion_terms = expander.expand(rewritten)
    else:
        expanded, expansion_terms = rewritten, []

    # 3. Metadata Filtering
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

    # 4. Hybrid Retrieval
    hybrid_service = HybridRetrievalService()
    if allowed_doc_ids is not None and len(allowed_doc_ids) == 0:
        hybrid_candidates = []
    elif allowed_doc_ids and len(allowed_doc_ids) > 1:
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

    if request.metadata_filter and hybrid_candidates:
        filter_svc = MetadataFilterService()
        hybrid_candidates = filter_svc.apply_to_chunks(hybrid_candidates, meta_filter)

    # 5. Reranking
    reranker = RerankerService()
    reranked = reranker.rerank(
        query=rewritten,
        candidates=hybrid_candidates,
        top_k=request.top_k,
    )

    # 6. Parent Document Retrieval
    if request.parent_window > 0 and reranked:
        parent_svc = ParentDocumentService(window_size=request.parent_window)
        final_chunks = parent_svc.fetch_parent_context(db=db, chunks=reranked)
    else:
        final_chunks = reranked

    # 7. Primary Document Grounded Answer Generation
    generation_svc = GenerationService()
    gen_result = generation_svc.generate_answer(query=rewritten, chunks=final_chunks)

    # 8. Confidence Scoring
    scorer = ConfidenceScorer(threshold=request.confidence_threshold)
    confidence_res = scorer.compute_confidence(
        retrieved_chunks=final_chunks,
        validation_result=gen_result["validation"],
        answer=gen_result["answer"],
    )

    # 9. Web Search Fallback if Low Confidence
    used_web_fallback = False
    web_details = None
    final_answer = gen_result["answer"]
    citations = gen_result["citations"]

    if confidence_res["is_low_confidence"] and request.enable_web_fallback:
        web_svc = WebSearchFallbackService()
        web_results = web_svc.search_web(request.query)
        web_gen = web_svc.generate_web_answer(request.query, web_results)

        used_web_fallback = True
        final_answer = f"[Web Fallback Used due to low document context confidence]\n\n{web_gen['answer']}"
        citations = web_gen["web_citations"]
        web_details = WebFallbackResultSchema(
            triggered=True,
            query=request.query,
            results_count=len(web_results),
            web_sources=[
                WebSourceItemSchema(
                    title=item["title"],
                    url=item["url"],
                    citation_tag=item["citation_tag"],
                )
                for item in web_gen["web_citations"]
            ],
        )

    return ReliableRAGResponse(
        query=request.query,
        guardrail=GuardrailStatusSchema(**guardrail_res),
        confidence=ConfidenceScoreSchema(**confidence_res),
        used_web_fallback=used_web_fallback,
        web_search_details=web_details,
        query_preprocessing=QueryPreprocessResult(
            original_query=request.query,
            rewritten_query=rewritten,
            expanded_query=expanded,
            expansion_terms=expansion_terms,
        ),
        retrieved_chunks=final_chunks,
        answer=final_answer,
        citations=citations,
        validation=gen_result["validation"],
    )
