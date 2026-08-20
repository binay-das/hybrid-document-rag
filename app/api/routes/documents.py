import io
import uuid
from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.document import Document, DocumentStatus, Page
from app.schemas.document import DocumentDetailResponse, DocumentResponse, PageResponse
from app.services.pdf import PDFParserService
from app.services.storage import StorageService

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/", response_model=DocumentDetailResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    if not (file.filename and file.filename.lower().endswith(".pdf")) and file.content_type != "application/pdf":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported",
        )

    content = await file.read()
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty",
        )

    file_size = len(content)
    storage_key = f"documents/{uuid.uuid4()}.pdf"
    storage = StorageService()
    storage.ensure_bucket()

    doc = Document(
        filename=storage_key.split("/")[-1],
        original_filename=file.filename or "document.pdf",
        mime_type=file.content_type or "application/pdf",
        file_size=file_size,
        storage_key=storage_key,
        bucket=storage.bucket,
        page_count=0,
        status=DocumentStatus.UPLOADING,
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    try:
        storage.upload_file(
            io.BytesIO(content),
            storage_key,
            file.content_type or "application/pdf",
        )

        doc.status = DocumentStatus.PROCESSING
        db.commit()

        parsed = PDFParserService.parse_pdf(content)

        doc.title = parsed.metadata.title
        doc.author = parsed.metadata.author
        doc.page_count = parsed.metadata.page_count
        doc.status = DocumentStatus.READY

        pages = [
            Page(
                document_id=doc.id,
                page_number=p.page_number,
                text=p.text,
            )
            for p in parsed.pages
        ]
        db.add_all(pages)
        db.commit()
        db.refresh(doc)
        return doc

    except Exception as e:
        db.rollback()
        doc.status = DocumentStatus.FAILED
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document processing failed: {str(e)}",
        )


@router.get("/", response_model=List[DocumentResponse])
def list_documents(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
):
    documents = db.query(Document).offset(skip).limit(limit).all()
    return documents


@router.get("/{document_id}", response_model=DocumentDetailResponse)
def get_document(
    document_id: int,
    db: Session = Depends(get_db),
):
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )
    return doc


@router.get("/{document_id}/pages", response_model=List[PageResponse])
def get_document_pages(
    document_id: int,
    db: Session = Depends(get_db),
):
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )
    return doc.pages


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(
    document_id: int,
    db: Session = Depends(get_db),
):
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    storage = StorageService()
    storage.delete_file(doc.storage_key)

    db.delete(doc)
    db.commit()