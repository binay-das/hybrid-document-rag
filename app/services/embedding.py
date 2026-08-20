import logging
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

        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not configured")

        self.client = genai.Client(api_key=self.api_key)

    def embed_text(self, text: str) -> List[float]:
        if not text.strip():
            raise ValueError("Cannot embed empty text")

        try:
            result = self.client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=text,
                config=types.EmbedContentConfig(
                    output_dimensionality=self.dimension,
                ),
            )

            return result.embeddings[0].values

        except Exception:
            logger.exception("Failed to generate embedding")
            raise

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        texts = [text for text in texts if text.strip()]

        if not texts:
            return []

        try:
            result = self.client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=texts,
                config=types.EmbedContentConfig(
                    output_dimensionality=self.dimension,
                ),
            )

            return [embedding.values for embedding in result.embeddings]

        except Exception:
            logger.exception("Failed to generate batch embeddings")
            raise