from fastapi import FastAPI

from app.api.routes.documents import router as documents_router
from app.services.storage import StorageService


app = FastAPI(title="Hybrid Document RAG")

app.include_router(documents_router)


@app.get("/health")
def health():
    storage = StorageService()
    storage.ensure_bucket()

    return {"status": "ok", "bucket": storage.bucket}