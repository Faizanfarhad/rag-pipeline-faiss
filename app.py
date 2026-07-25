"""
FastAPI RAG Service
===================
Exposes the RagPipeline as an HTTP endpoint.
Config via environment variables, no hardcoded secrets.
Logs per-query latency, chunk count, and token usage.
"""

import os
import time
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ═══════════════════════════════════════════════
# Configuration via environment variables
# ═══════════════════════════════════════════════

DOC_PATH = os.getenv("RAG_DOC_PATH", "docsContainer/text.md")
TOP_K = int(os.getenv("RAG_TOP_K", "10"))
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "50"))
EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DEVICE = os.getenv("RAG_DEVICE", "cpu")

# ── Logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("rag-api")

# ── Pipeline singleton ──
pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load pipeline on startup, clean up on shutdown."""
    global pipeline
    from RagPipeline.rag_pipeline_connector import RagPipelineConnector

    logger.info(f"Loading pipeline for: {DOC_PATH}")
    t0 = time.time()
    pipeline = RagPipelineConnector(
        doc_path=DOC_PATH,
        top_k=TOP_K,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        embd_model=EMBED_MODEL,
        device=DEVICE,
    )
    logger.info(f"Pipeline ready in {time.time() - t0:.1f}s")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="RAG QA Service",
    description="Cost-efficient RAG API backed by FAISS",
    version="1.0.0",
    lifespan=lifespan,
)


# ═══════════════════════════════════════════════
# Request / Response models
# ═══════════════════════════════════════════════

class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = None


class SourceChunk(BaseModel):
    id: int
    text: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    source_chunks: list[SourceChunk]
    total_sources: int
    latency_ms: float
    top_k: int


class HealthResponse(BaseModel):
    status: str
    doc_path: str
    top_k: int
    chunk_size: int
    chunk_overlap: int
    embed_model: str
    device: str


# ═══════════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════════

@app.get("/health", response_model=HealthResponse)
async def health():
    """Return pipeline configuration."""
    return HealthResponse(
        status="ready" if pipeline else "loading",
        doc_path=DOC_PATH,
        top_k=TOP_K,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        embed_model=EMBED_MODEL,
        device=DEVICE,
    )


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """Ask a question against the document."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded yet")

    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    t_start = time.time()
    try:
        result = pipeline.query(req.question)
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    latency_ms = (time.time() - t_start) * 1000

    # Log per-query metrics
    logger.info(
        f"Q: '{req.question[:80]}...' | "
        f"latency: {latency_ms:.0f}ms | "
        f"chunks: {result['total_sources']}"
    )

    return QueryResponse(
        question=req.question,
        answer=result["answer"],
        source_chunks=[
            SourceChunk(id=c["id"], text=c["text"]) for c in result["source_chunks"]
        ],
        total_sources=result["total_sources"],
        latency_ms=round(latency_ms, 2),
        top_k=req.top_k or TOP_K,
    )


@app.get("/query")
async def query_get(q: str = Query(..., description="Question to ask")):
    """GET endpoint for simple browser queries."""
    return await query(QueryRequest(question=q))


# ═══════════════════════════════════════════════
# Run: uvicorn app:app --host 0.0.0.0 --port 8000
# ═══════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)