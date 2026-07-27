import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
from datasets import Dataset as HFDataset

class SentencePairDataset(Dataset):
    def __init__(self, hf_dataset: HFDataset, is_regression: bool = False):
        self.dataset = hf_dataset
        self.is_regression = is_regression

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[str, str, float]:
        item = self.dataset[idx]
        label = float(item['label']) if self.is_regression else int(item['label'])
        return str(item['text_a']), str(item['text_b']), label


def collate_sentence_pairs(batch: List[Tuple[str, str, float]]) -> Tuple[List[str], List[str], torch.Tensor]:
    texts_a = [item[0] for item in batch]
    texts_b = [item[1] for item in batch]
    labels = [item[2] for item in batch]
    
    if isinstance(labels[0], float):
        labels_tensor = torch.tensor(labels, dtype=torch.float32)
    else:
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        
    return texts_a, texts_b, labels_tensor


def get_dataloader(
    hf_dataset: HFDataset, 
    batch_size: int, 
    is_regression: bool = False, 
    shuffle: bool = True,
    num_workers: int = 4
) -> DataLoader:
  
    dataset = SentencePairDataset(hf_dataset, is_regression=is_regression)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_sentence_pairs,
        num_workers=num_workers,
        pin_memory=True
    )