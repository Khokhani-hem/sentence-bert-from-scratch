import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin

    def forward(
        self, 
        embeddings_a: torch.Tensor, 
        embeddings_b: torch.Tensor, 
        labels: torch.Tensor
    ) -> torch.Tensor:
  
        distances = F.pairwise_distance(embeddings_a, embeddings_b, keepdim=True)
        
        labels = labels.view(-1, 1).float()
        
        losses = 0.5 * (
            labels * torch.pow(distances, 2) +
            (1 - labels) * torch.pow(torch.clamp(self.margin - distances, min=0.0), 2)
        )
        
        return losses.mean()