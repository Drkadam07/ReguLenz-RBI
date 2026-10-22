import faiss
import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np

class FAISSVectorStore:
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner Product for normalized vectors
        self.metadata: List[Dict[str, Any]] = []

    def add(self, embeddings: np.ndarray, documents: List[Dict[str, Any]]):
        self.index.add(embeddings)
        self.metadata.extend(documents)

    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[Dict[str, Any], float]]:
        scores, indices = self.index.search(query_embedding, top_k)
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx != -1 and idx < len(self.metadata):
                results.append((self.metadata[idx], float(score)))
        return results

    def save(self, directory: Path):
        faiss.write_index(self.index, str(directory / "index.faiss"))
        with open(directory / "metadata.pkl", "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self, directory: Path):
        self.index = faiss.read_index(str(directory / "index.faiss"))
        with open(directory / "metadata.pkl", "rb") as f:
            self.metadata = pickle.load(f)