import torch
import torch.nn as nn
from typing import Dict


class Pooling(nn.Module):
    def __init__(self, pooling_strategy: str = "mean"):
        super().__init__()
        self.pooling_strategy = pooling_strategy.lower()
        
        valid_strategies = {"mean", "max", "cls"}
        if self.pooling_strategy not in valid_strategies:
            raise ValueError(f"Invalid pooling_strategy '{pooling_strategy}'. Must be one of {valid_strategies}")

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        token_embeddings = features["token_embeddings"]
        attention_mask = features["attention_mask"]

        if self.pooling_strategy == "cls":
            sentence_embedding = token_embeddings[:, 0, :]
            
        elif self.pooling_strategy == "mean":
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            sentence_embedding = sum_embeddings / sum_mask
            
        elif self.pooling_strategy == "max":
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            
            masked_embeddings = token_embeddings.clone()
            masked_embeddings[input_mask_expanded == 0] = -1e9
            sentence_embedding = torch.max(masked_embeddings, 1)[0]
            
        else:
            raise NotImplementedError(f"Pooling strategy {self.pooling_strategy} not implemented.")

        features.update({"sentence_embedding": sentence_embedding})
        return features