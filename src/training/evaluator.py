import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from typing import Dict
import logging

from src.models.siamese import SiameseNetwork
from src.utils.metrics import compute_pearson_correlation, compute_spearman_correlation

logger = logging.getLogger(__name__)

class STSEvaluator:
    def __init__(self, dataloader: DataLoader, device: torch.device):
        self.dataloader = dataloader
        self.device = device

    @torch.no_grad()
    def __call__(self, model: SiameseNetwork) -> Dict[str, float]:
        model.eval()
        
        all_scores = []
        all_labels = []
        
        progress_bar = tqdm(self.dataloader, desc="Evaluating STS")
        
        for batch in progress_bar:
            texts_a, texts_b, labels = batch

            emb_a = model.encode(texts_a, self.device, batch_size=len(texts_a)).to(self.device)
            emb_b = model.encode(texts_b, self.device, batch_size=len(texts_b)).to(self.device)
            
            cosine_scores = F.cosine_similarity(emb_a, emb_b, dim=1)
            
            all_scores.extend(cosine_scores.cpu().numpy())
            all_labels.extend(labels.numpy())

        scores_np = np.array(all_scores)
        labels_np = np.array(all_labels)

        pearson = compute_pearson_correlation(labels_np, scores_np)
        spearman = compute_spearman_correlation(labels_np, scores_np)
        
        results = {
            "pearson": pearson,
            "spearman": spearman
        }
        
        logger.info(f"STS Evaluation - Pearson: {pearson:.4f} | Spearman: {spearman:.4f}")
        return results