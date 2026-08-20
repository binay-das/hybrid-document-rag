import uuid

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.services.storage import StorageService

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/")
async def upload_document(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    storage = StorageService()
    storage.ensure_bucket()

    key = f"documents/{uuid.uuid4()}.pdf"

    storage.upload_file(
        file.file,
        key,
        file.content_type,
    )

    return {
        "filename": file.filename,
        "storage_key": key,
        "status": "UPLOADED",
    }