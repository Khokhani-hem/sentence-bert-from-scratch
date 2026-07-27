import argparse
import yaml
import torch
import pandas as pd
from datasets import Dataset
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from pathlib import Path

from src.utils.seed import seed_everything
from src.utils.logger import get_logger
from src.data.preprocessing import prepare_stsb_dataset, prepare_nli_dataset
from src.data.loader import get_dataloader
from src.models.siamese import SiameseNetwork

from src.losses.cosine_loss import CosineSimilarityLoss
from src.losses.contrastive_loss import ContrastiveLoss
from src.losses.triplet_loss import TripletLoss
from src.losses.multiple_negative_ranking import MultipleNegativesRankingLoss
from src.training.trainer import SentenceBERTTrainer

def create_triplet_dataset(nli_dataset: Dataset) -> Dataset:
    df = nli_dataset.to_pandas()
    entailments = df[df['label'] == 0].rename(columns={'text_a': 'anchor', 'text_b': 'positive'})
    contradictions = df[df['label'] == 2].rename(columns={'text_a': 'anchor', 'text_b': 'negative'})
    
    triplets = pd.merge(entailments[['anchor', 'positive']], 
                        contradictions[['anchor', 'negative']], 
                        on='anchor').dropna()
    

    triplets = triplets.rename(columns={'anchor': 'text_a', 'positive': 'text_b', 'negative': 'label'})
    return Dataset.from_pandas(triplets)

def main():
    parser = argparse.ArgumentParser(description="Train Siamese Network for Sentence Embeddings")
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    parser.add_argument("--model_config", type=str, default="configs/model.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        train_config = yaml.safe_load(f)
    with open(args.model_config, "r") as f:
        model_config = yaml.safe_load(f)

    seed_everything(train_config["training"]["seed"])
    logger = get_logger(__name__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    loss_type = model_config["loss"]["type"].lower()
    logger.info(f"Connecting Loss Function: {loss_type.upper()}")

    if loss_type == "cosine":
        train_ds = prepare_stsb_dataset(split="train")
        is_regression = True
    else:
        train_ds = prepare_nli_dataset()
        is_regression = False
        
        if loss_type == "mnrl":

            train_ds = train_ds.filter(lambda x: x["label"] == 0)
        elif loss_type == "contrastive":
            train_ds = train_ds.filter(lambda x: x["label"] in [0, 2])
            def map_labels(example):
                example["label"] = 1 if example["label"] == 0 else 0
                return example
            train_ds = train_ds.map(map_labels)
        elif loss_type == "triplet":

            train_ds = create_triplet_dataset(train_ds)

    val_ds = prepare_stsb_dataset(split="validation")
    train_dl = get_dataloader(train_ds, batch_size=train_config["training"]["batch_size"], is_regression=is_regression)
    val_dl = get_dataloader(val_ds, batch_size=train_config["training"]["batch_size"], is_regression=True, shuffle=False)

    logger.info(f"Initializing Siamese Network with {model_config['model']['pooling_strategy']} pooling...")
    model = SiameseNetwork(
        model_name=model_config["model"]["backbone"],
        max_seq_length=model_config["model"]["max_seq_length"],
        pooling_strategy=model_config["model"]["pooling_strategy"]
    ).to(device)

    if loss_type == "cosine":
        loss_fn = CosineSimilarityLoss()
    elif loss_type == "contrastive":
        loss_fn = ContrastiveLoss(margin=model_config["loss"].get("margin", 0.5))
    elif loss_type == "triplet":
        loss_fn = TripletLoss(margin=model_config["loss"].get("margin", 1.0))
    elif loss_type == "mnrl":
        loss_fn = MultipleNegativesRankingLoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


    optimizer = AdamW(
        model.parameters(), 
        lr=float(train_config["training"]["learning_rate"]),
        weight_decay=train_config["training"]["weight_decay"]
    )
    
    total_steps = len(train_dl) * train_config["training"]["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=train_config["training"]["warmup_steps"],
        num_training_steps=total_steps
    )

    exp_output_dir = Path(train_config["paths"]["output_dir"]) / f"{loss_type}_{model_config['model']['pooling_strategy']}"
    train_config["training"]["output_dir"] = str(exp_output_dir)

    trainer = SentenceBERTTrainer(
        model=model,
        train_dataloader=train_dl,
        val_dataloader=val_dl,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        config=train_config["training"],
        device=device
    )

    trainer.train()
    logger.info(f"Experiment finished. Best model saved to {exp_output_dir}")

if __name__ == "__main__":
    main()