import torch
import torch.nn as nn

class MultipleNegativesRankingLoss(nn.Module):
    def __init__(self, scale: float = 20.0):
        super().__init__()
        self.scale = scale
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(
        self, 
        embeddings_a: torch.Tensor, 
        embeddings_b: torch.Tensor
    ) -> torch.Tensor:
        norm_a = nn.functional.normalize(embeddings_a, p=2, dim=1)
        norm_b = nn.functional.normalize(embeddings_b, p=2, dim=1)
        
        scores = torch.mm(norm_a, norm_b.transpose(0, 1)) * self.scale
        
        labels = torch.arange(scores.size(0), device=scores.device)
        
        return self.cross_entropy(scores, labels)