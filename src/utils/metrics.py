import numpy as np
from scipy.stats import pearsonr, spearmanr
import torch

def compute_pearson_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return 0.0
    corr, _ = pearsonr(y_true, y_pred)
    return float(corr)

def compute_spearman_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return 0.0
    corr, _ = spearmanr(y_true, y_pred)
    return float(corr)

def compute_top_k_accuracy(
    similarity_matrix: torch.Tensor, 
    target_indices: torch.Tensor, 
    k: int = 5
) -> float:

    _, top_k_indices = torch.topk(similarity_matrix, k, dim=1)
    
    correct = 0
    for i, target in enumerate(target_indices):
        if target in top_k_indices[i]:
            correct += 1
            
    return correct / len(target_indices)