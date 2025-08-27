from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

class LocalEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en"):
        self.model = SentenceTransformer(model_name)

    def embed(self, text: str) -> np.ndarray:
        vec = self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return vec

    def embed_many(self, texts: List[str]) -> List[np.ndarray]:
        vecs = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return vecs

# ========== CLI test ==========

if __name__ == "__main__":
    print("\n[Test] Embedder")
    embedder = LocalEmbedder()
    text = "This is a test sentence for embedding."
    vector = embedder.embed(text)
    print("Vector shape:", vector.shape)
    print("First 5 dims:", vector[:5])
    
    

