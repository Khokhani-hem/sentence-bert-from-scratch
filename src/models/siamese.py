import torch
import torch.nn as nn
from typing import Union, List
from .bert_encoder import BertEncoder
from .pooling import Pooling


class SiameseNetwork(nn.Module):
    def __init__(
        self, 
        model_name: str = "bert-base-uncased", 
        max_seq_length: int = 128, 
        pooling_strategy: str = "mean",
        freeze_backbone: bool = False
    ):

        super().__init__()
        self.encoder = BertEncoder(
            model_name=model_name, 
            max_seq_length=max_seq_length,
            freeze_backbone=freeze_backbone
        )
        self.pooling = Pooling(pooling_strategy=pooling_strategy)

    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> torch.Tensor:

        features = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_features = self.pooling(features)
        
        return pooled_features["sentence_embedding"]

    @torch.no_grad()
    def encode(
        self, 
        texts: Union[str, List[str]], 
        device: torch.device,
        batch_size: int = 32
    ) -> torch.Tensor:
        
        self.eval()
        if isinstance(texts, str):
            texts = [texts]
            
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = self.encoder.tokenize(batch_texts, device)
            
            embeddings = self.forward(
                input_ids=inputs["input_ids"], 
                attention_mask=inputs["attention_mask"]
            )
            all_embeddings.append(embeddings.cpu())
            
        return torch.cat(all_embeddings, dim=0)