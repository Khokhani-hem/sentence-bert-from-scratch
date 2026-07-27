import faiss
import numpy as np
import logging

logger = logging.getLogger(__name__)

class FAISSIndex:
    def __init__(self, embedding_dim: int = 768):

        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatIP(embedding_dim)
        
    def add(self, embeddings: np.ndarray):
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(f"Expected embedding dimension {self.embedding_dim}, got {embeddings.shape[1]}")
            
        embeddings = embeddings.astype(np.float32)
        
        faiss.normalize_L2(embeddings)
        
        self.index.add(embeddings)
        logger.info(f"Added {len(embeddings)} vectors. Total in index: {self.index.ntotal}")

    def search(self, query_embeddings: np.ndarray, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
        if self.index.ntotal == 0:
            raise RuntimeError("Cannot search an empty index. Add vectors first.")
            
        query_embeddings = query_embeddings.astype(np.float32)
        faiss.normalize_L2(query_embeddings)
        
        distances, indices = self.index.search(query_embeddings, k)
        return distances, indices