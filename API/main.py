from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from pydantic import BaseModel, Field
from typing import List, Optional
from contextlib import asynccontextmanager
from pathlib import Path
import shutil

from src.config import settings
from src.ingestion.parser import RegulatoryPDFParser
from src.ingestion.chunker import RegulatoryChunker
from src.retrieval.embeddings import EmbeddingModel
from src.retrieval.vector_store import FAISSVectorStore
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.hybrid_retriever import HybridRetriever
from src.generation.prompt import build_rag_prompt
from src.generation.llm import OllamaLLMClient
from src.evaluation.retrieval_eval import run_evaluation

# Global state containers
state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize components on startup
    state["parser"] = RegulatoryPDFParser()
    state["chunker"] = RegulatoryChunker()
    state["embeddings"] = EmbeddingModel()
    state["reranker"] = CrossEncoderReranker()
    state["llm"] = OllamaLLMClient()
    state["vector_store"] = FAISSVectorStore()

    # Load stored index if present
    if (settings.DATA_PROCESSED_DIR / "index.faiss").exists():
        state["vector_store"].load(settings.DATA_PROCESSED_DIR)
        corpus = state["vector_store"].metadata
        state["retriever"] = HybridRetriever(
            vector_store=state["vector_store"],
            embedding_model=state["embeddings"],
            reranker=state["reranker"],
            corpus=corpus,
        )
    else:
        state["retriever"] = None
    yield
    state.clear()


app = FastAPI(title="RBI Circulars RAG Assistant API", lifespan=lifespan)


class QueryRequest(BaseModel):
    question: str = Field(
        ..., example="What are the KYC updation timelines for high risk accounts?"
    )
    top_k: int = Field(default=3, ge=1, le=10)


class SourceCitation(BaseModel):
    document_id: str
    page: int
    chunk_id: str
    rerank_score: float
    snippet: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceCitation]


@app.get("/health")
def health_check():
    return {"status": "ok", "index_loaded": state.get("retriever") is not None}


@app.post("/ingest")
async def ingest_file(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF documents are supported.")

    saved_path = settings.DATA_RAW_DIR / file.filename
    with open(saved_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    elements = state["parser"].parse(saved_path)
    chunks = state["chunker"].chunk_document(elements)

    if not chunks:
        raise HTTPException(status_code=400, detail="Document could not be chunked.")

    texts = [c["content"] for c in chunks]
    embeddings = state["embeddings"].encode(texts)

    state["vector_store"].add(embeddings, chunks)
    state["vector_store"].save(settings.DATA_PROCESSED_DIR)

    # Re-initialize hybrid retriever with updated corpus
    state["retriever"] = HybridRetriever(
        vector_store=state["vector_store"],
        embedding_model=state["embeddings"],
        reranker=state["reranker"],
        corpus=state["vector_store"].metadata,
    )

    return {
        "message": f"Successfully ingested {file.filename}",
        "chunks_indexed": len(chunks),
    }


@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    if not state.get("retriever"):
        raise HTTPException(
            status_code=503, detail="Search index is not built. Ingest documents first."
        )

    retrieved = state["retriever"].search(
        query=request.question, top_k=settings.TOP_K_RETRIEVAL, final_k=request.top_k
    )

    if not retrieved:
        return QueryResponse(
            answer="I couldn't find sufficient evidence in the provided circulars.",
            sources=[],
        )

    prompt = build_rag_prompt(request.question, retrieved)
    answer = await state["llm"].generate_answer(prompt)

    sources = [
        SourceCitation(
            document_id=doc["document_id"],
            page=doc["page"],
            chunk_id=doc["chunk_id"],
            rerank_score=doc.get("rerank_score", 0.0),
            snippet=doc["content"][:200] + "...",
        )
        for doc in retrieved
    ]

    return QueryResponse(answer=answer, sources=sources)


@app.get("/evaluate")
def run_eval():
    if not state.get("retriever"):
        raise HTTPException(status_code=503, detail="Search index is not built.")
    if not settings.EVAL_DATA_PATH.exists():
        raise HTTPException(status_code=404, detail="Benchmark dataset not found.")

    def search_fn(q: str, top_k: int):
        return state["retriever"].search(query=q, top_k=top_k, final_k=top_k)

    results = run_evaluation(settings.EVAL_DATA_PATH, search_fn)
    return {"metrics": results}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app", host=settings.API_HOST, port=settings.API_PORT, reload=True
    )
