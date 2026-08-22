import re
import logging
from typing import Any, Dict, List, Optional
from google import genai
from google.genai import types

from app.core.config import settings

logger = logging.getLogger(__name__)

GENERATION_MODEL = "gemini-2.5-flash"


class GenerationService:
    """
    RAG Answer Generation & Citation Validation Service.
    Generates strictly grounded answers from reranked context chunks using Gemini
    and validates page-level document citations.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.gemini_api_key
        self.client = None
        if self.api_key and self.api_key.strip():
            try:
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                logger.warning(f"Could not initialize Gemini Client for Generation: {e}")

    def format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Formats retrieved context chunks with explicit document and page citations.
        """
        context_blocks = []
        for idx, chunk in enumerate(chunks, start=1):
            doc_id = chunk.get("document_id")
            page_num = chunk.get("page_number", 1)
            text = chunk.get("text", "").strip()
            block = f"[Source #{idx} | Doc #{doc_id}, Page {page_num}]\n{text}"
            context_blocks.append(block)
        return "\n\n".join(context_blocks)

    def validate_citations(
        self,
        answer: str,
        retrieved_chunks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Validates page-level citations in the generated answer against retrieved context chunks.
        """
        # Valid page set from retrieved chunks: (document_id, page_number)
        retrieved_pages = {
            (c.get("document_id"), c.get("page_number"))
            for c in retrieved_chunks
        }

        # Extract [Doc #X, Page Y] patterns from answer
        citation_matches = re.findall(r"\[Doc\s*#(\d+),\s*Page\s*(\d+)\]", answer, re.IGNORECASE)

        valid_citations = []
        invalid_citations = []
        cited_pages = []

        for doc_str, page_str in citation_matches:
            doc_id = int(doc_str)
            page_num = int(page_str)
            page_tuple = (doc_id, page_num)
            
            citation_item = {"document_id": doc_id, "page_number": page_num}
            cited_pages.append(citation_item)

            if page_tuple in retrieved_pages:
                valid_citations.append(citation_item)
            else:
                invalid_citations.append(citation_item)

        total_citations = len(citation_matches)
        if total_citations == 0:
            grounding_score = 1.0
            is_valid = True
        else:
            grounding_score = len(valid_citations) / total_citations
            is_valid = len(invalid_citations) == 0

        return {
            "is_valid": is_valid,
            "grounding_score": round(grounding_score, 4),
            "total_citations": total_citations,
            "valid_citations": valid_citations,
            "invalid_citations": invalid_citations,
            "cited_pages": cited_pages,
        }

    def generate_answer(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Generates a grounded answer with page-level citations for a query using retrieved context.
        """
        if not chunks:
            answer = "I cannot answer this question based on the provided context."
            return {
                "answer": answer,
                "citations": [],
                "validation": self.validate_citations(answer, []),
                "context_used_count": 0,
            }

        context_str = self.format_context(chunks)

        prompt = (
            "You are a precise RAG assistant. Answer the question relying strictly and ONLY on the provided context.\n"
            "CRITICAL RULES:\n"
            "1. Do NOT use outside knowledge.\n"
            "2. If the context does not contain enough information to answer, reply EXACTLY: 'I cannot answer this question based on the provided context.'\n"
            "3. For every statement or claim, you MUST cite the source using the exact format: [Doc #<document_id>, Page <page_number>].\n\n"
            f"CONTEXT:\n{context_str}\n\n"
            f"QUESTION: {query}\n\n"
            "ANSWER:"
        )

        answer_text = ""
        if self.client:
            try:
                response = self.client.models.generate_content(
                    model=GENERATION_MODEL,
                    contents=prompt,
                )
                answer_text = response.text.strip() if response.text else ""
            except Exception as e:
                logger.warning(f"Gemini generation API call failed ({e}). Falling back to deterministic grounded response.")

        if not answer_text:
            # Deterministic fallback answer constructed strictly from top retrieved chunk
            top_chunk = chunks[0]
            doc_id = top_chunk.get("document_id")
            page_num = top_chunk.get("page_number", 1)
            text_snippet = top_chunk.get("text", "").strip()
            answer_text = f"Based on the context: {text_snippet[:250]}... [Doc #{doc_id}, Page {page_num}]"

        validation = self.validate_citations(answer_text, chunks)

        return {
            "answer": answer_text,
            "citations": validation["cited_pages"],
            "validation": validation,
            "context_used_count": len(chunks),
        }
