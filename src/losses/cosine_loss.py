import torch
import torch.nn as nn

class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.MSELoss()

    def forward(
        self, 
        embeddings_a: torch.Tensor, 
        embeddings_b: torch.Tensor, 
        labels: torch.Tensor
    ) -> torch.Tensor:
    
        scores = torch.cosine_similarity(embeddings_a, embeddings_b, dim=1)
        
        labels = labels.view(-1).float()
        
        loss = self.loss_fn(scores, labels)
        
        return loss