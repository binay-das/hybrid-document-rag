from fastapi import FastAPI

app = FastAPI(title="Hybrid Document RAG")


@app.get("/health")
def health():
    return {"status": "ok"}