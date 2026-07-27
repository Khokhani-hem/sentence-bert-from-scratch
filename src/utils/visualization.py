import os
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from typing import List, Optional
import logging

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

logger = logging.getLogger(__name__)

class EmbeddingVisualizer:
    def __init__(self, method: str = 'umap', random_state: int = 42):
        self.method = method.lower()
        self.random_state = random_state
        
        if self.method == 'pca':
            self.reducer = PCA(n_components=2, random_state=random_state)
        elif self.method == 'tsne':
            self.reducer = TSNE(n_components=2, random_state=random_state, perplexity=30)
        elif self.method == 'umap':
            self.reducer = umap.UMAP(n_components=2, random_state=random_state, metric='cosine')
        else:
            raise ValueError("Method must be one of: 'pca', 'tsne', 'umap'")

    def _reduce_dimensions(self, embeddings: np.ndarray) -> np.ndarray:
        logger.info(f"Reducing dimensions using {self.method.upper()}...")

        if self.method == 'tsne' and len(embeddings) < 30:
            self.reducer.set_params(perplexity=max(1, len(embeddings) - 1))
            
        return self.reducer.fit_transform(embeddings)

    def plot_static(
        self, 
        embeddings: np.ndarray, 
        labels: np.ndarray, 
        title: str = "Embedding Space", 
        save_path: Optional[str] = None
    ):
        reduced_embeddings = self._reduce_dimensions(embeddings)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            reduced_embeddings[:, 0], 
            reduced_embeddings[:, 1], 
            c=labels, 
            cmap='viridis', 
            alpha=0.7,
            edgecolors='w',
            linewidth=0.5
        )
        plt.colorbar(scatter, label='Classes / Similarity')
        plt.title(f"{title} ({self.method.upper()})")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.grid(True, linestyle='--', alpha=0.3)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved static plot to {save_path}")
        else:
            plt.show()
            
        plt.close()

    def plot_interactive(
        self, 
        embeddings: np.ndarray, 
        texts: List[str], 
        labels: np.ndarray, 
        title: str = "Interactive Embedding Space",
        save_path: Optional[str] = None
    ):
        reduced_embeddings = self._reduce_dimensions(embeddings)
        
        fig = px.scatter(
            x=reduced_embeddings[:, 0], 
            y=reduced_embeddings[:, 1], 
            color=labels,
            hover_name=texts,  # FIXED: Plotly accepts lists here without needing a DataFrame
            title=f"{title} ({self.method.upper()})",
            color_continuous_scale="Viridis",
            opacity=0.8
        )
        
        fig.update_layout(
            xaxis_title="Component 1",
            yaxis_title="Component 2",
            template="plotly_white"
        )
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.write_html(save_path)
            logger.info(f"Saved interactive plot to {save_path}")
        else:
            fig.show()