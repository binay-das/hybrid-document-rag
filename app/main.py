from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.advanced_retrieval import router as advanced_router
from app.api.routes.documents import router as documents_router
from app.api.routes.generation import router as generation_router
from app.api.routes.retrieval import router as retrieval_router
from app.api.routes.reliability import router as reliability_router
from app.services.storage import StorageService

app = FastAPI(title="Hybrid Document RAG")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents_router)
app.include_router(retrieval_router)
app.include_router(generation_router)
app.include_router(advanced_router)
app.include_router(reliability_router)



@app.get("/health")
def health():
    storage = StorageService()
    storage.ensure_bucket()

    return {"status": "ok", "bucket": storage.bucket}