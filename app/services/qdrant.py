import logging
from typing import Any, Dict, List

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams

from app.core.config import settings
from app.models.document import Chunk

logger = logging.getLogger(__name__)

DEFAULT_COLLECTION_NAME = "chunks"
DEFAULT_VECTOR_SIZE = 768


class QdrantService:
    def __init__(self):
        self.client = QdrantClient(url=settings.qdrant_url)
        self.collection_name = DEFAULT_COLLECTION_NAME

    def ensure_collection(
        self,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        vector_size: int = DEFAULT_VECTOR_SIZE,
    ):
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == collection_name for c in collections)
            if not exists:
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info(f"Created Qdrant collection: {collection_name} (dim={vector_size})")
        except Exception as e:
            logger.error(f"Error ensuring Qdrant collection {collection_name}: {e}")
            raise

    def upsert_chunks(
        self,
        chunks: List[Chunk],
        embeddings: List[List[float]],
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ):
        if not chunks or not embeddings:
            return

        self.ensure_collection(collection_name=collection_name)

        points = [
            PointStruct(
                id=chunk.id,
                vector=embedding,
                payload={
                    "chunk_id": chunk.id,
                    "document_id": chunk.document_id,
                    "page_id": chunk.page_id,
                    "page_number": chunk.page_number,
                    "chunk_index": chunk.chunk_index,
                    "text": chunk.text,
                    "char_count": chunk.char_count,
                },
            )
            for chunk, embedding in zip(chunks, embeddings)
        ]

        self.client.upsert(
            collection_name=collection_name,
            points=points,
        )
        logger.info(f"Upserted {len(points)} vectors into Qdrant collection '{collection_name}'")

    def delete_chunks_by_document(
        self,
        document_id: int,
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ):
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == collection_name for c in collections)
            if not exists:
                return

            self.client.delete(
                collection_name=collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=document_id),
                        )
                    ]
                ),
            )
            logger.info(f"Deleted Qdrant vectors for document_id={document_id}")
        except Exception as e:
            logger.error(f"Error deleting vectors for document_id={document_id}: {e}")

    def count_chunks(self, collection_name: str = DEFAULT_COLLECTION_NAME) -> int:
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == collection_name for c in collections)
            if not exists:
                return 0
            return self.client.count(collection_name=collection_name).count
        except Exception as e:
            logger.error(f"Error counting vectors in Qdrant: {e}")
            return 0

    def search_vectors(
        self,
        query_vector: List[float],
        top_k: int = 5,
        document_id: int | None = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ) -> List[Dict[str, Any]]:
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == collection_name for c in collections)
            if not exists:
                return []

            query_filter = None
            if document_id is not None:
                query_filter = Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=document_id),
                        )
                    ]
                )

            res = self.client.query_points(
                collection_name=collection_name,
                query=query_vector,
                query_filter=query_filter,
                limit=top_k,
            )

            hits = res.points if hasattr(res, "points") else res
            results = []
            for hit in hits:
                payload = hit.payload or {}
                results.append({
                    "chunk_id": payload.get("chunk_id", hit.id),
                    "document_id": payload.get("document_id"),
                    "page_id": payload.get("page_id"),
                    "page_number": payload.get("page_number"),
                    "chunk_index": payload.get("chunk_index"),
                    "text": payload.get("text", ""),
                    "char_count": payload.get("char_count", 0),
                    "score": float(hit.score),
                })
            return results
        except Exception as e:
            logger.error(f"Error executing vector search in Qdrant: {e}")
            return []


