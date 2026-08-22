import hashlib
import logging
import math
from typing import List

from google import genai
from google.genai import types

from app.core.config import settings

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "gemini-embedding-001"
VECTOR_DIMENSION = 768


class EmbeddingService:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or settings.gemini_api_key
        self.dimension = VECTOR_DIMENSION

        self.client = None
        if self.api_key and self.api_key.strip():
            try:
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                logger.warning(f"Could not initialize Gemini Client: {e}")

    def _generate_mock_vector(self, text: str) -> List[float]:
        """Generate a deterministic 768-dimensional unit vector from text hash."""
        vec = []
        for i in range(self.dimension):
            seed = f"{text}_{i}".encode("utf-8")
            h = hashlib.sha256(seed).hexdigest()
            val = (int(h[:8], 16) / 0xFFFFFFFF) * 2.0 - 1.0
            vec.append(val)

        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    def embed_text(self, text: str) -> List[float]:
        if not text or not text.strip():
            return [0.0] * self.dimension

        if not self.client:
            logger.info("Gemini client uninitialized / API key empty. Using mock fallback embedding.")
            return self._generate_mock_vector(text)

        try:
            result = self.client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=text,
                config=types.EmbedContentConfig(
                    output_dimensionality=self.dimension,
                ),
            )
            return result.embeddings[0].values
        except Exception as e:
            logger.warning(f"Gemini API call failed ({e}). Falling back to mock vector.")
            return self._generate_mock_vector(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        valid_texts = [text for text in texts if text and text.strip()]
        if not valid_texts:
            return []

        if not self.client:
            logger.info("Gemini client uninitialized / API key empty. Using mock fallback batch embeddings.")
            return [self._generate_mock_vector(t) for t in valid_texts]

        try:
            result = self.client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=valid_texts,
                config=types.EmbedContentConfig(
                    output_dimensionality=self.dimension,
                ),
            )
            return [embedding.values for embedding in result.embeddings]
        except Exception as e:
            logger.warning(f"Gemini Batch API call failed ({e}). Falling back to mock vectors.")
            return [self._generate_mock_vector(t) for t in valid_texts]