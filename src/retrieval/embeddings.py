from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List
from src.config import settings

class EmbeddingModel:
    def __init__(self, model_name: str = settings.EMBEDDING_MODEL_NAME):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        # BGE models benefit from normalized vectors for cosine similarity
        embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.array(embeddings, dtype=np.float32)