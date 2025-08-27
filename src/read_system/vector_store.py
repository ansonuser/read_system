import faiss
import pickle
import numpy as np
from typing import List, Dict, Tuple

class VectorStore:
    def __init__(self, dim: int, index_path: str = None):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # cosine similarity (if vectors are normalized)
        self.metadata: List[Dict] = []
        self.index_path = index_path

    def add(self, vectors: List[np.ndarray], metadatas: List[Dict]):
        stacked = np.stack(vectors)
        self.index.add(stacked)
        self.metadata.extend(metadatas)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[float, Dict]]:
        query_vector = np.expand_dims(query_vector, axis=0)
        scores, indices = self.index.search(query_vector, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.metadata):
                results.append((float(score), self.metadata[idx]))
        return results

    def save(self, index_file: str, metadata_file: str):
        faiss.write_index(self.index, index_file)
        with open(metadata_file, "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self, index_file: str, metadata_file: str):
        self.index = faiss.read_index(index_file)
        with open(metadata_file, "rb") as f:
            self.metadata = pickle.load(f)