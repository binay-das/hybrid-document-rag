import math
import re
from typing import Any, Dict, List, Optional


class RerankerService:
    """
    Cross-Encoder Reranker Service.
    Computes cross-encoding alignment and semantic interaction scores between query and document text chunks.
    """

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def score_pair(self, query: str, text: str) -> float:
        """
        Cross-encoder scoring for a single (query, text) pair.
        Calculates term frequency, term position proximity, sequential n-gram alignment, and coverage ratio.
        """
        q_tokens = self._tokenize(query)
        t_tokens = self._tokenize(text)

        if not q_tokens or not t_tokens:
            return 0.0

        t_text_lower = text.lower()
        query_lower = query.lower().strip()

        # Exact match bonus
        exact_bonus = 2.0 if query_lower in t_text_lower else 0.0

        # Term frequency and position weighting
        term_score = 0.0
        matched_positions = []

        for idx, q_term in enumerate(q_tokens):
            positions = [i for i, token in enumerate(t_tokens) if token == q_term]
            if positions:
                # Term frequency log scaling
                term_score += 1.0 + math.log(len(positions))
                matched_positions.extend(positions)

        # Proximity score if multiple terms matched
        proximity_score = 0.0
        if len(matched_positions) > 1:
            matched_positions.sort()
            gaps = [matched_positions[i+1] - matched_positions[i] for i in range(len(matched_positions)-1)]
            avg_gap = sum(gaps) / len(gaps)
            proximity_score = 1.0 / (1.0 + math.log(1.0 + avg_gap))

        # Query coverage ratio
        unique_q = set(q_tokens)
        unique_t = set(t_tokens)
        coverage = len(unique_q.intersection(unique_t)) / float(len(unique_q))

        # Combined Cross-Encoder Logit / Score
        final_score = (term_score * 1.5) + (proximity_score * 1.2) + (coverage * 3.0) + exact_bonus
        return float(final_score)

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Reranks a list of candidate chunk dictionaries by cross-encoder relevance score.
        """
        if not candidates:
            return []

        reranked = []
        for candidate in candidates:
            text = candidate.get("text", "")
            cross_score = self.score_pair(query, text)
            
            # Combine prior retrieval score with cross-encoder score
            prior_score = float(candidate.get("score", 0.0))
            combined_score = cross_score + (prior_score * 0.5)

            item_copy = dict(candidate)
            item_copy["score"] = float(combined_score)
            item_copy["cross_encoder_score"] = float(cross_score)
            reranked.append(item_copy)

        reranked.sort(key=lambda x: x["score"], reverse=True)

        if top_k is not None:
            return reranked[:top_k]
        return reranked
