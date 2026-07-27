import torch
import logging
from typing import List, Dict, Union

from src.models.siamese import SiameseNetwork
from .faiss_index import FAISSIndex

logger = logging.getLogger(__name__)

class SemanticSearch:
    def __init__(self, model: SiameseNetwork, device: torch.device):
        self.model = model
        self.model.eval()
        self.device = device
        
        self.faiss_index = None
        self.corpus = []

    def build_index(self, corpus: List[str], batch_size: int = 32):
        logger.info(f"Encoding corpus of {len(corpus)} documents...")
        self.corpus = corpus
        
        embeddings = self.model.encode(corpus, self.device, batch_size=batch_size)
        embeddings_np = embeddings.cpu().numpy()
        
        embedding_dim = embeddings_np.shape[1]
        self.faiss_index = FAISSIndex(embedding_dim=embedding_dim)

        self.faiss_index.add(embeddings_np)
        logger.info("Index build complete.")

    def search(self, query: Union[str, List[str]], k: int = 5) -> List[List[Dict]]:
        if self.faiss_index is None:
            raise RuntimeError("Index not built. Call build_index(corpus) first.")
            
        if isinstance(query, str):
            query = [query]
            
        # Encode the queries
        query_embeddings = self.model.encode(query, self.device).cpu().numpy()
        
        k = min(k, len(self.corpus))
        scores, indices = self.faiss_index.search(query_embeddings, k)
        
        results = []
        for i in range(len(query)):
            query_results = []
            for j in range(k):
                doc_idx = indices[i][j]
                query_results.append({
                    "score": float(scores[i][j]),
                    "text": self.corpus[doc_idx]
                })
            results.append(query_results)
            
        return results