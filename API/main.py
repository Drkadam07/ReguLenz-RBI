import os
import shutil
from pathlib import Path
from typing import List, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from pydantic import BaseModel, Field

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

# Global state tracker
state = {
    "is_indexing": False,
    "indexing_progress": "Idle",
    "retriever": None
}

def load_or_reload_retriever():
    """Hot-loads the hybrid retriever from disk without restarting FastAPI."""
    index_file = settings.DATA_PROCESSED_DIR / "index.faiss"
    metadata_file = settings.DATA_PROCESSED_DIR / "metadata.pkl"
    
    if index_file.exists() and metadata_file.exists():
        vstore = FAISSVectorStore()
        vstore.load(settings.DATA_PROCESSED_DIR)
        
        state["vector_store"] = vstore
        state["retriever"] = HybridRetriever(
            vector_store=vstore,
            embedding_model=state["embeddings"],
            reranker=state["reranker"],
            corpus=vstore.metadata
        )
        return True
    return False

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Setup models at startup
    state["parser"] = RegulatoryPDFParser()
    state["chunker"] = RegulatoryChunker()
    state["embeddings"] = EmbeddingModel()
    state["reranker"] = CrossEncoderReranker()
    state["llm"] = OllamaLLMClient()
    
    load_or_reload_retriever()
    yield
    state.clear()

app = FastAPI(title="RBI Circulars RAG Assistant API", lifespan=lifespan)

class QueryRequest(BaseModel):
    question: str = Field(..., example="What are the KYC updation timelines for high risk accounts?")
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
    has_retriever = state.get("retriever") is not None
    chunk_count = len(state["retriever"].corpus) if has_retriever else 0
    return {
        "status": "online",
        "index_loaded": has_retriever,
        "total_chunks_indexed": chunk_count,
        "is_indexing_in_progress": state["is_indexing"]
    }

def background_bulk_index_task():
    try:
        state["is_indexing"] = True
        state["indexing_progress"] = "Parsing PDFs in data/raw..."
        
        pdf_files = list(settings.DATA_RAW_DIR.glob("*.pdf"))
        if not pdf_files:
            state["indexing_progress"] = "No PDFs found in data/raw"
            state["is_indexing"] = False
            return

        all_chunks = []
        for pdf in pdf_files:
            state["indexing_progress"] = f"Parsing {pdf.name}..."
            elements = state["parser"].parse(pdf)
            chunks = state["chunker"].chunk_document(elements)
            all_chunks.extend(chunks)

        if not all_chunks:
            state["indexing_progress"] = "Extraction produced 0 chunks."
            state["is_indexing"] = False
            return

        state["indexing_progress"] = f"Embedding {len(all_chunks)} chunks..."
        texts = [c["content"] for c in all_chunks]
        embeddings = state["embeddings"].encode(texts)

        state["indexing_progress"] = "Saving index to disk..."
        vstore = FAISSVectorStore(dimension=embeddings.shape[1])
        vstore.add(embeddings, all_chunks)
        vstore.save(settings.DATA_PROCESSED_DIR)

        # Hot-reload retriever in memory
        load_or_reload_retriever()
        state["indexing_progress"] = f"Completed! Indexed {len(all_chunks)} chunks."
    except Exception as e:
        state["indexing_progress"] = f"Failed: {str(e)}"
    finally:
        state["is_indexing"] = False

@app.post("/build-index-all")
def trigger_bulk_indexing(background_tasks: BackgroundTasks):
    """Triggers background processing of all PDFs inside data/raw/."""
    if state["is_indexing"]:
        raise HTTPException(status_code=409, detail="An indexing job is already active.")
    
    background_tasks.add_task(background_bulk_index_task)
    return {"message": "Bulk indexing started in background."}

@app.get("/index-status")
def get_index_status():
    return {
        "is_indexing": state["is_indexing"],
        "status_message": state["indexing_progress"],
        "index_ready": state.get("retriever") is not None
    }

@app.post("/ingest")
async def ingest_single_file(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF documents are supported.")
    
    dest_path = settings.DATA_RAW_DIR / file.filename
    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    elements = state["parser"].parse(dest_path)
    chunks = state["chunker"].chunk_document(elements)
    
    if not chunks:
        raise HTTPException(status_code=400, detail="No readable text extracted from PDF.")

    texts = [c["content"] for c in chunks]
    embeddings = state["embeddings"].encode(texts)
    
    # Initialize store or append
    if state.get("vector_store") is None:
        state["vector_store"] = FAISSVectorStore(dimension=embeddings.shape[1])
        
    state["vector_store"].add(embeddings, chunks)
    state["vector_store"].save(settings.DATA_PROCESSED_DIR)

    load_or_reload_retriever()
    return {"message": f"Successfully indexed {file.filename}", "chunks_added": len(chunks)}

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    if not state.get("retriever"):
        raise HTTPException(status_code=503, detail="Search index is not built. Ingest documents first.")

    retrieved = state["retriever"].search(
        query=request.question, 
        top_k=settings.TOP_K_RETRIEVAL, 
        final_k=request.top_k
    )
    
    if not retrieved:
        return QueryResponse(
            answer="I couldn't find sufficient evidence in the provided circulars.",
            sources=[]
        )

    prompt = build_rag_prompt(request.question, retrieved)
    answer = await state["llm"].generate_answer(prompt)

    sources = [
        SourceCitation(
            document_id=doc["document_id"],
            page=doc["page"],
            chunk_id=doc["chunk_id"],
            rerank_score=doc.get("rerank_score", 0.0),
            snippet=doc["content"][:220] + "..."
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