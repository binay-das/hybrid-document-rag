from datetime import datetime
from pydantic import BaseModel, ConfigDict
from app.models.document import DocumentStatus


class PageResponse(BaseModel):
    id: int
    document_id: int
    page_number: int
    text: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ChunkResponse(BaseModel):
    id: int
    document_id: int
    page_id: int | None = None
    chunk_index: int
    page_number: int
    text: str
    char_count: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ChunkingRequest(BaseModel):
    chunk_size: int = 500
    chunk_overlap: int = 50


class ChunkingSummaryResponse(BaseModel):
    document_id: int
    total_chunks: int
    chunk_size: int
    chunk_overlap: int
    chunks: list[ChunkResponse]


class DocumentResponse(BaseModel):
    id: int
    filename: str
    original_filename: str
    mime_type: str
    file_size: int
    storage_key: str
    bucket: str
    title: str | None = None
    author: str | None = None
    page_count: int
    status: DocumentStatus
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class DocumentDetailResponse(DocumentResponse):
    pages: list[PageResponse] = []
    chunks: list[ChunkResponse] = []

    model_config = ConfigDict(from_attributes=True)