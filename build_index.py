import sys
from pathlib import Path
from tqdm import tqdm
import numpy as np

from src.config import settings
from src.ingestion.parser import RegulatoryPDFParser
from src.ingestion.chunker import RegulatoryChunker
from src.retrieval.embeddings import EmbeddingModel
from src.retrieval.vector_store import FAISSVectorStore

def run_bulk_indexing(batch_size: int = 64):
    print("=" * 60)
    print("RBI Circulars RAG - Offline Bulk Indexer")
    print("=" * 60)

    pdf_files = list(settings.DATA_RAW_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"[-] No PDF files found in {settings.DATA_RAW_DIR}")
        print("    Place your circular PDFs inside data/raw/ and re-run.")
        return False

    print(f"[+] Found {len(pdf_files)} PDF(s) to process.")
    parser = RegulatoryPDFParser()
    chunker = RegulatoryChunker()
    
    all_chunks = []
    for pdf_path in tqdm(pdf_files, desc="Parsing Documents"):
        elements = parser.parse(pdf_path)
        chunks = chunker.chunk_document(elements)
        all_chunks.extend(chunks)

    if not all_chunks:
        print("[-] No chunks extracted. Verify your PDF contents.")
        return False

    print(f"[+] Created {len(all_chunks)} text chunks across all documents.")
    print("[+] Loading embedding model into memory...")
    embedder = EmbeddingModel()

    print(f"[+] Generating dense embeddings in batches of {batch_size}...")
    texts = [c["content"] for c in all_chunks]
    
    # Process in batches to prevent CPU/GPU memory saturation
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding Chunks"):
        batch_texts = texts[i : i + batch_size]
        emb_batch = embedder.encode(batch_texts)
        all_embeddings.append(emb_batch)

    final_embeddings = np.vstack(all_embeddings)

    print("[+] Building and persisting FAISS vector index...")
    # Get embedding dimension dynamically
    dim = final_embeddings.shape[1]
    store = FAISSVectorStore(dimension=dim)
    store.add(final_embeddings, all_chunks)
    store.save(settings.DATA_PROCESSED_DIR)

    print(f"[✔] Indexing completed successfully!")
    print(f"    Index files saved at: {settings.DATA_PROCESSED_DIR.resolve()}")
    return True

if __name__ == "__main__":
    success = run_bulk_indexing()
    if not success:
        sys.exit(1)