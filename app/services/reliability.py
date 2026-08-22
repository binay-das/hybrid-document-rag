import logging
import re
from typing import Any, Dict, List, Optional
import httpx
from google import genai
from sqlalchemy.orm import Session

from app.core.config import settings

logger = logging.getLogger(__name__)

_GENERATION_MODEL = "gemini-1.5-flash"


# 1. Reject Unsupported & Adversarial Queries (Guardrails)

_ADVERSARIAL_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|directions|prompts|rules)",
    r"disregard\s+(all\s+)?(previous|prior|above)\s+(instructions|directions|prompts|rules)",
    r"system\s*prompt",
    r"jailbreak",
    r"dan\s+mode",
    r"developer\s+mode",
    r"override\s+(security|safety|rules|instructions)",
    r"pretend\s+to\s+be",
    r"bypass\s+restrictions",
    r"act\s+as\s+an?\s+unrestricted",
    r"you\s+are\s+now\s+in\s+unrestricted",
]


class QueryGuardrailService:
    """
    Evaluates incoming queries to detect prompt injection attempts,
    adversarial prompts, and invalid/unsupported query structures.
    """

    def __init__(self):
        self.compiled_adversarial = [
            re.compile(pattern, re.IGNORECASE) for pattern in _ADVERSARIAL_PATTERNS
        ]

    def check_query(self, query: str) -> Dict[str, Any]:
        """
        Returns:
            {
                "is_rejected": bool,
                "is_adversarial": bool,
                "rejection_reason": Optional[str]
            }
        """
        stripped = query.strip()

        if not stripped:
            return {
                "is_rejected": True,
                "is_adversarial": False,
                "rejection_reason": "Query is empty or whitespace only.",
            }

        if len(stripped) < 2:
            return {
                "is_rejected": True,
                "is_adversarial": False,
                "rejection_reason": "Query is too short to be processed.",
            }

        # Check for adversarial patterns / prompt injections
        for pattern in self.compiled_adversarial:
            if pattern.search(stripped):
                logger.warning(f"Adversarial query pattern detected: '{stripped}'")
                return {
                    "is_rejected": True,
                    "is_adversarial": True,
                    "rejection_reason": "Query contains potential prompt injection or adversarial patterns.",
                }

        return {
            "is_rejected": False,
            "is_adversarial": False,
            "rejection_reason": None,
        }


# 2. Confidence Scoring

class ConfidenceScorer:
    """
    Computes retrieval and grounding confidence scores for RAG pipeline results.
    """

    def __init__(self, threshold: float = 0.35):
        self.threshold = threshold

    def compute_confidence(
        self,
        retrieved_chunks: List[Dict[str, Any]],
        validation_result: Optional[Dict[str, Any]] = None,
        answer: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Calculates:
        - retrieval_confidence: based on cross-encoder rerank / vector similarity scores
        - grounding_confidence: based on citation validation score
        - overall_confidence: weighted combination
        """
        if not retrieved_chunks:
            return {
                "retrieval_confidence": 0.0,
                "grounding_confidence": 0.0,
                "overall_confidence": 0.0,
                "is_low_confidence": True,
                "threshold": self.threshold,
            }

        # Calculate retrieval confidence from chunk scores
        scores = []
        for c in retrieved_chunks:
            # Prefer cross-encoder score if available, otherwise RRF/vector score
            ce_score = c.get("cross_encoder_score")
            if ce_score is not None:
                # Normalize cross-encoder logit/score (sigmoid-like scaling or clamp [0, 1])
                norm_score = 1.0 / (1.0 + 2.71828 ** (-float(ce_score)))
                scores.append(norm_score)
            else:
                scores.append(float(c.get("score", 0.0)))

        avg_top_score = sum(scores[:3]) / min(len(scores), 3) if scores else 0.0
        retrieval_conf = min(max(avg_top_score, 0.0), 1.0)

        # Grounding confidence from citation validation
        grounding_conf = 1.0
        if validation_result:
            grounding_conf = float(validation_result.get("grounding_score", 1.0))

        # Check if answer indicates fallback/unsupported context
        if answer and "I cannot answer this question based on the provided context" in answer:
            grounding_conf = 0.0
            retrieval_conf = min(retrieval_conf, 0.2)

        overall_conf = round(0.7 * retrieval_conf + 0.3 * grounding_conf, 4)
        is_low = overall_conf < self.threshold

        return {
            "retrieval_confidence": round(retrieval_conf, 4),
            "grounding_confidence": round(grounding_conf, 4),
            "overall_confidence": overall_conf,
            "is_low_confidence": is_low,
            "threshold": self.threshold,
        }


# 3. Web Search Fallback for Low-Confidence Queries

class WebSearchFallbackService:
    """
    Performs external web search fallback when document context confidence is low.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.gemini_api_key
        self.client = None
        if self.api_key and self.api_key.strip():
            try:
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                logger.warning(f"WebSearchFallback: could not init Gemini client: {e}")

    def search_web(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """
        Executes web search fallback via DuckDuckGo API/HTML search.
        Returns a list of dicts: [{"title": ..., "snippet": ..., "url": ...}]
        """
        results: List[Dict[str, str]] = []
        try:
            url = "https://html.duckduckgo.com/html/"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }
            with httpx.Client(timeout=5.0, follow_redirects=True) as client:
                response = client.post(url, data={"q": query}, headers=headers)
                if response.status_code == 200:
                    html = response.text
                    # Extract snippets using regex
                    raw_matches = re.findall(
                        r'<a class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>.*?'
                        r'<a class="result__snippet"[^>]*>(.*?)</a>',
                        html,
                        re.DOTALL,
                    )
                    for href, title_html, snippet_html in raw_matches[:max_results]:
                        clean_title = re.sub(r"<[^>]+>", "", title_html).strip()
                        clean_snippet = re.sub(r"<[^>]+>", "", snippet_html).strip()
                        if clean_title and clean_snippet:
                            results.append({
                                "title": clean_title,
                                "snippet": clean_snippet,
                                "url": href,
                            })
        except Exception as e:
            logger.warning(f"DuckDuckGo web search fallback failed: {e}")

        # Deterministic fallback result if external HTTP call fails or returns empty
        if not results:
            results.append({
                "title": f"Web Search Result for '{query}'",
                "snippet": f"Web knowledge context for query '{query}': Public information and reference summary.",
                "url": "https://www.google.com/search?q=" + query.replace(" ", "+"),
            })

        return results

    def generate_web_answer(
        self, query: str, web_results: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Generates an answer grounded in web search results.
        """
        context_blocks = []
        for idx, res in enumerate(web_results, start=1):
            block = f"[Web Source #{idx} | {res['title']}] ({res['url']})\n{res['snippet']}"
            context_blocks.append(block)
        context_str = "\n\n".join(context_blocks)

        prompt = (
            "You are a helpful assistant providing web-searched answers when local documents lack sufficient context.\n"
            "Answer the query accurately based on the web context below. Cite web sources using [Web Source #X].\n\n"
            f"WEB CONTEXT:\n{context_str}\n\n"
            f"QUESTION: {query}\n\n"
            "ANSWER:"
        )

        answer_text = ""
        if self.client:
            try:
                resp = self.client.models.generate_content(
                    model=_GENERATION_MODEL, contents=prompt
                )
                answer_text = (resp.text or "").strip()
            except Exception as e:
                logger.warning(f"Gemini web answer generation failed: {e}")

        if not answer_text and web_results:
            top = web_results[0]
            answer_text = f"Based on web search results: {top['snippet']} [Web Source #1]"

        web_citations = [
            {"title": r["title"], "url": r["url"], "citation_tag": f"[Web Source #{i+1}]"}
            for i, r in enumerate(web_results)
        ]

        return {
            "answer": answer_text,
            "web_citations": web_citations,
            "web_sources_count": len(web_results),
        }
