import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Dict, List, Union

class BertEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        max_seq_length: int = 128,
        freeze_backbone: bool = False
    ):
        super().__init__()
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        
        if freeze_backbone:
            for param in self.bert.parameters():
                param.requires_grad = False

    def tokenize(
        self, 
        texts: Union[str, List[str]], 
        device: torch.device
    ) -> Dict[str, torch.Tensor]:
    
        if isinstance(texts, str):
            texts = [texts]
            
        encoding = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt"
        )
        
        return {key: val.to(device) for key, val in encoding.items()}

    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        token_embeddings = outputs.last_hidden_state
        
        return {
            "token_embeddings": token_embeddings,
            "attention_mask": attention_mask
        }