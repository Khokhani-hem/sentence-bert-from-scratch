import torch
import torch.nn as nn

class TripletLoss(nn.Module):

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.loss_fn = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(
        self, 
        anchor: torch.Tensor, 
        positive: torch.Tensor, 
        negative: torch.Tensor
    ) -> torch.Tensor:
  
        return self.loss_fn(anchor, positive, negative)