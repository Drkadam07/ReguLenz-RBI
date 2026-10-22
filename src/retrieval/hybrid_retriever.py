from rank_bm25 import BM25Okapi
from typing import List, Dict, Any
from src.retrieval.vector_store import FAISSVectorStore
from src.retrieval.embeddings import EmbeddingModel
from src.retrieval.reranker import CrossEncoderReranker

class HybridRetriever:
    """Combines BM25 lexical search with FAISS dense retrieval via Reciprocal Rank Fusion."""
    def __init__(
        self, 
        vector_store: FAISSVectorStore, 
        embedding_model: EmbeddingModel, 
        reranker: CrossEncoderReranker,
        corpus: List[Dict[str, Any]],
        rrf_constant: int = 60
    ):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.reranker = reranker
        self.corpus = corpus
        self.rrf_k = rrf_constant
        
        # Build BM25 index
        self.tokenized_corpus = [doc["content"].lower().split() for doc in self.corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str, top_k: int = 10, final_k: int = 3) -> List[Dict[str, Any]]:
        # 1. Lexical BM25 Search
        query_tokens = query.lower().split()
        bm25_scores = self.bm25.get_scores(query_tokens)
        bm25_ranked_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k]

        # 2. Dense Semantic Search
        q_emb = self.embedding_model.encode([query])
        dense_results = self.vector_store.search(q_emb, top_k=top_k)

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        doc_lookup = {}

        # Rank aggregation from BM25
        for rank, idx in enumerate(bm25_ranked_indices):
            doc = self.corpus[idx]
            cid = doc["chunk_id"]
            doc_lookup[cid] = doc
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + (rank + 1)))

        # Rank aggregation from Dense
        for rank, (doc, _) in enumerate(dense_results):
            cid = doc["chunk_id"]
            doc_lookup[cid] = doc
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + (rank + 1)))

        # Sort RRF candidates
        fused_cids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)[:top_k]
        candidates = [doc_lookup[cid] for cid in fused_cids]

        # 4. Neural Cross-Encoder Reranking
        return self.reranker.rerank(query=query, documents=candidates, top_k=final_k)