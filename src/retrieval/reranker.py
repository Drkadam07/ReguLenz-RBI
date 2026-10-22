from sentence_transformers import CrossEncoder
from typing import List, Dict, Any
from src.config import settings

class CrossEncoderReranker:
    def __init__(self, model_name: str = settings.RERANKER_MODEL_NAME):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
        if not documents:
            return []
        
        pairs = [[query, doc["content"]] for doc in documents]
        scores = self.model.predict(pairs)
        
        for doc, score in zip(documents, scores):
            doc["rerank_score"] = float(score)

        ranked = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)
        return ranked[:top_k]