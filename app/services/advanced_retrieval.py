import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from google import genai
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.document import Chunk, Document

logger = logging.getLogger(__name__)
_GENERATION_MODEL = "gemini-2.5-flash"


# Query Rewriting
class QueryRewritingService:
    """
    Rewrites a user query into a cleaner, more retrieval-optimised form.
    Uses Gemini when available; deterministic heuristic fallback otherwise.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.gemini_api_key
        self.client = None
        if self.api_key and self.api_key.strip():
            try:
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                logger.warning(f"QueryRewriting: could not init Gemini client: {e}")

    def rewrite(self, query: str) -> str:
        """Return a single improved retrieval query for the original query."""
        if not query.strip():
            return query

        if self.client:
            prompt = (
                "You are a search query optimizer. Rewrite the following user query into a "
                "concise, keyword-rich search query that maximises retrieval recall from a "
                "document database. Output ONLY the rewritten query—no explanation.\n\n"
                f"Original query: {query}\n\nRewritten query:"
            )
            try:
                resp = self.client.models.generate_content(
                    model=_GENERATION_MODEL, contents=prompt
                )
                rewritten = (resp.text or "").strip().strip('"').strip("'")
                if rewritten:
                    logger.info(f"Query rewritten: '{query}' → '{rewritten}'")
                    return rewritten
            except Exception as e:
                logger.warning(f"QueryRewriting Gemini call failed ({e}). Using fallback.")

        # Heuristic fallback: remove filler words, lowercase, deduplicate tokens
        fillers = {"what", "is", "the", "a", "an", "of", "in", "on", "at", "to",
                   "for", "with", "about", "can", "you", "tell", "me", "does", "do",
                   "how", "are", "was", "were", "has", "have", "been", "be", "that",
                   "this", "please", "explain"}
        tokens = re.findall(r"\w+", query.lower())
        seen = set()
        kept = []
        for t in tokens:
            if t not in fillers and t not in seen:
                seen.add(t)
                kept.append(t)
        rewritten = " ".join(kept) if kept else query
        logger.info(f"Query rewritten (heuristic): '{query}' → '{rewritten}'")
        return rewritten


# Query Expansion
# Static synonym map — augmented by Gemini when available
_SYNONYM_MAP: Dict[str, List[str]] = {
    "author": ["writer", "creator", "composer"],
    "title": ["name", "heading", "caption"],
    "summary": ["abstract", "overview", "synopsis"],
    "method": ["approach", "technique", "algorithm", "procedure"],
    "result": ["outcome", "finding", "conclusion"],
    "dataset": ["data", "corpus", "collection", "benchmark"],
    "performance": ["accuracy", "metric", "evaluation", "benchmark"],
    "model": ["architecture", "network", "system"],
    "training": ["learning", "fine-tuning", "optimization"],
    "introduction": ["background", "overview", "preface"],
    "table": ["chart", "figure", "grid"],
    "page": ["section", "chapter", "part"],
    "document": ["paper", "article", "report", "file"],
    "error": ["bug", "issue", "failure", "fault"],
    "image": ["picture", "figure", "diagram", "illustration"],
}


class QueryExpansionService:
    """
    Expands a query with semantically related terms / synonyms.
    Uses the static synonym map first, then optionally calls Gemini
    to generate additional expansion terms.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.gemini_api_key
        self.client = None
        if self.api_key and self.api_key.strip():
            try:
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                logger.warning(f"QueryExpansion: could not init Gemini client: {e}")

    def _static_expansion(self, query: str) -> List[str]:
        tokens = re.findall(r"\w+", query.lower())
        extra: List[str] = []
        for token in tokens:
            synonyms = _SYNONYM_MAP.get(token, [])
            for s in synonyms:
                if s.lower() not in query.lower():
                    extra.append(s)
        return extra

    def _gemini_expansion(self, query: str) -> List[str]:
        if not self.client:
            return []
        prompt = (
            "List 5 short synonyms or related search terms for the following query. "
            "Output ONLY a comma-separated list of terms, no explanation.\n\n"
            f"Query: {query}\n\nTerms:"
        )
        try:
            resp = self.client.models.generate_content(
                model=_GENERATION_MODEL, contents=prompt
            )
            raw = (resp.text or "").strip()
            terms = [t.strip().strip("'\"") for t in raw.split(",") if t.strip()]
            return terms[:5]
        except Exception as e:
            logger.warning(f"QueryExpansion Gemini call failed ({e}).")
            return []

    def expand(self, query: str) -> Tuple[str, List[str]]:
        """
        Returns (expanded_query, expansion_terms).
        expanded_query appends extra terms to the original for broader recall.
        """
        static_terms = self._static_expansion(query)
        gemini_terms = self._gemini_expansion(query)
        all_extra = list(dict.fromkeys(static_terms + gemini_terms))  # deduplicate, preserve order
        if all_extra:
            expanded = f"{query} {' '.join(all_extra)}"
        else:
            expanded = query
        logger.info(f"Query expanded: '{query}' → extra terms: {all_extra}")
        return expanded, all_extra


# Metadata Filtering

class MetadataFilter:
    """Structured metadata filter applied before retrieval."""

    def __init__(
        self,
        document_ids: Optional[List[int]] = None,
        author: Optional[str] = None,
        title: Optional[str] = None,
        min_page: Optional[int] = None,
        max_page: Optional[int] = None,
    ):
        self.document_ids = document_ids
        self.author = author
        self.title = title
        self.min_page = min_page
        self.max_page = max_page


class MetadataFilterService:
    """
    Resolves a MetadataFilter against the Postgres documents table,
    producing a final list of allowed document_ids and a Qdrant-compatible filter dict.
    """

    def resolve_document_ids(
        self, db: Session, meta_filter: MetadataFilter
    ) -> Optional[List[int]]:
        """
        Returns a flat list of allowed document_ids after applying metadata constraints.
        Returns None if no constraint (i.e. all documents allowed).
        """
        query = db.query(Document.id)

        if meta_filter.document_ids:
            query = query.filter(Document.id.in_(meta_filter.document_ids))

        if meta_filter.author:
            query = query.filter(
                Document.author.ilike(f"%{meta_filter.author}%")
            )

        if meta_filter.title:
            query = query.filter(
                Document.title.ilike(f"%{meta_filter.title}%")
            )

        doc_ids = [row[0] for row in query.all()]

        # If caller applied filters but nothing matched, return empty list (no results)
        if (
            meta_filter.document_ids
            or meta_filter.author
            or meta_filter.title
        ) and not doc_ids:
            return []

        return doc_ids if doc_ids else None

    def apply_to_chunks(
        self, chunks: List[Dict[str, Any]], meta_filter: MetadataFilter
    ) -> List[Dict[str, Any]]:
        """Post-filter chunk dicts by page range (applied after retrieval)."""
        if meta_filter.min_page is None and meta_filter.max_page is None:
            return chunks

        filtered = []
        for c in chunks:
            pn = c.get("page_number", 0)
            if meta_filter.min_page is not None and pn < meta_filter.min_page:
                continue
            if meta_filter.max_page is not None and pn > meta_filter.max_page:
                continue
            filtered.append(c)
        return filtered


# Parent-Document Retrieval

class ParentDocumentService:
    """
    Given a set of retrieved leaf chunks, fetches their 'parent' context window:
    the immediately adjacent chunks (prev + next) from the same document,
    then merges them into an expanded context block.
    """

    def __init__(self, window_size: int = 1):
        """
        window_size: number of sibling chunks to include on each side.
        1 → prev + self + next (3 chunks per result)
        2 → 2 prev + self + 2 next (5 chunks per result)
        """
        self.window_size = window_size

    def fetch_parent_context(
        self,
        db: Session,
        chunks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        For each retrieved chunk, loads window_size siblings on each side.
        Returns a deduplicated, ordered list of expanded context dicts.
        """
        if not chunks:
            return []

        # Collect (document_id, chunk_index) windows to load
        needed: Dict[int, set] = {}  # doc_id → set of chunk_indices
        for chunk in chunks:
            doc_id = chunk["document_id"]
            ci = chunk["chunk_index"]
            if doc_id not in needed:
                needed[doc_id] = set()
            for offset in range(-self.window_size, self.window_size + 1):
                idx = ci + offset
                if idx >= 0:
                    needed[doc_id].add(idx)

        # Batch-load all needed chunks from DB
        sibling_map: Dict[Tuple[int, int], Chunk] = {}
        for doc_id, indices in needed.items():
            rows = (
                db.query(Chunk)
                .filter(
                    Chunk.document_id == doc_id,
                    Chunk.chunk_index.in_(sorted(indices)),
                )
                .all()
            )
            for row in rows:
                sibling_map[(row.document_id, row.chunk_index)] = row

        # Build expanded results, preserving original scores for retrieved chunks
        score_map: Dict[int, float] = {c["chunk_id"]: c["score"] for c in chunks}
        ce_map: Dict[int, float] = {
            c["chunk_id"]: c.get("cross_encoder_score", 0.0)
            for c in chunks
            if c.get("cross_encoder_score") is not None
        }

        seen_ids: set = set()
        expanded: List[Dict[str, Any]] = []

        # Process in original retrieval order so primary chunks come first
        for chunk in chunks:
            doc_id = chunk["document_id"]
            ci = chunk["chunk_index"]
            for offset in range(-self.window_size, self.window_size + 1):
                sibling = sibling_map.get((doc_id, ci + offset))
                if sibling is None or sibling.id in seen_ids:
                    continue
                seen_ids.add(sibling.id)

                is_primary = sibling.id == chunk["chunk_id"]
                expanded.append({
                    "chunk_id": sibling.id,
                    "document_id": sibling.document_id,
                    "page_id": sibling.page_id,
                    "page_number": sibling.page_number,
                    "chunk_index": sibling.chunk_index,
                    "text": sibling.text,
                    "char_count": sibling.char_count,
                    "score": score_map.get(sibling.id, 0.0),
                    "cross_encoder_score": ce_map.get(sibling.id),
                    "is_primary": is_primary,
                })

        return expanded
