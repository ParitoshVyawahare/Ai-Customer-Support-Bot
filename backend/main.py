"""FastAPI app exposing /upload, /ask, /clear, /health."""
import shutil
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from document_loader import load_documents
from rag_engine import RAGEngine

# ---------- Models ----------

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The user's question")
    session_id: str = Field("default", description="Conversation session ID")


class SourceModel(BaseModel):
    id: int
    source: str
    chunk_id: int
    score: float
    cited: bool
    snippet: str


class AskResponse(BaseModel):
    answer: str
    sources: List[SourceModel]
    confidence: str


class ClearRequest(BaseModel):
    session_id: str = "default"


class HealthResponse(BaseModel):
    status: str
    doc_count: int
    model: str


class UploadResponse(BaseModel):
    files: List[str]
    chunks_added: int
    total_chunks_in_db: int


# ---------- App ----------

app = FastAPI(
    title="AI Customer Support Bot",
    description="RAG-powered customer support with multi-query retrieval, source citations, and conversation memory.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG engine on startup (heavy: loads embedding model)
rag: Optional[RAGEngine] = None


@app.on_event("startup")
def _startup():
    global rag
    rag = RAGEngine()


# ---------- Endpoints ----------

@app.get("/health", response_model=HealthResponse)
def health():
    from config import settings
    return HealthResponse(
        status="healthy",
        doc_count=rag.get_doc_count() if rag else 0,
        model=settings.LLM_MODEL,
    )


@app.post("/upload", response_model=UploadResponse)
async def upload(files: List[UploadFile] = File(...)):
    """Upload PDF / TXT / CSV files. They are chunked, embedded, and stored in ChromaDB."""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    upload_dir = Path("./uploads")
    upload_dir.mkdir(exist_ok=True)

    allowed_ext = {".pdf", ".txt", ".csv"}
    saved_paths: List[Path] = []

    for file in files:
        ext = Path(file.filename).suffix.lower()
        if ext not in allowed_ext:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file.filename}. Allowed: {sorted(allowed_ext)}",
            )
        dest = upload_dir / file.filename
        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)
        saved_paths.append(dest)

    total_added = 0
    for path in saved_paths:
        try:
            chunks = load_documents(str(path))
            added = rag.add_documents(chunks)
            total_added += added
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to process {path.name}: {e}",
            )

    return UploadResponse(
        files=[p.name for p in saved_paths],
        chunks_added=total_added,
        total_chunks_in_db=rag.get_doc_count(),
    )


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """Ask a question. Returns an answer with cited sources and a confidence label."""
    if rag.get_doc_count() == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents in the knowledge base. Upload documents via /upload first.",
        )
    result = rag.query(req.question, session_id=req.session_id)
    return AskResponse(**result)


@app.post("/clear")
def clear(req: ClearRequest):
    """Clear conversation memory for a session."""
    rag.clear_memory(req.session_id)
    return {"status": "cleared", "session_id": req.session_id}


# Optional: inspect what's in memory (useful for debugging)
@app.get("/memory/{session_id}")
def get_memory(session_id: str):
    return {"session_id": session_id, "messages": rag.get_memory(session_id)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)