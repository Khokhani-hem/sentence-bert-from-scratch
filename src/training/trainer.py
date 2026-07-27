import torch
import os
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

class SentenceBERTTrainer:
    def __init__(
        self, 
        model, 
        train_dataloader: DataLoader, 
        val_dataloader: DataLoader, 
        loss_fn, 
        optimizer, 
        scheduler, 
        config: dict, 
        device: torch.device
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs = config.get("epochs", 4)
        self.output_dir = config.get("output_dir", "outputs/")
        self.device = device
        self.max_grad_norm = config.get("max_grad_norm", 1.0)
        self.is_triplet = loss_fn.__class__.__name__ == "TripletLoss"

    def train(self):
        os.makedirs(self.output_dir, exist_ok=True)
        best_loss = float('inf')

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            progress = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{self.epochs}")

            for batch in progress:
                self.optimizer.zero_grad()
                
                if self.is_triplet:

                    texts_a, texts_p, texts_n = batch
                    
                    inputs_a = self.model.encoder.tokenize(texts_a, self.device)
                    inputs_p = self.model.encoder.tokenize(texts_p, self.device)
                    inputs_n = self.model.encoder.tokenize(texts_n, self.device)
                    
                    emb_a = self.model(inputs_a["input_ids"], inputs_a["attention_mask"])
                    emb_p = self.model(inputs_p["input_ids"], inputs_p["attention_mask"])
                    emb_n = self.model(inputs_n["input_ids"], inputs_n["attention_mask"])
                    
                    loss = self.loss_fn(emb_a, emb_p, emb_n)
                    
                else:

                    texts_a, texts_b, labels = batch
                    labels = labels.to(self.device)
                    
                    inputs_a = self.model.encoder.tokenize(texts_a, self.device)
                    inputs_b = self.model.encoder.tokenize(texts_b, self.device)

                    emb_a = self.model(inputs_a["input_ids"], inputs_a["attention_mask"])
                    emb_b = self.model(inputs_b["input_ids"], inputs_b["attention_mask"])


                    if self.loss_fn.__class__.__name__ == "MultipleNegativesRankingLoss":
                        loss = self.loss_fn(emb_a, emb_b)
                    else:
                        loss = self.loss_fn(emb_a, emb_b, labels)
                    
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                self.optimizer.step()
                self.scheduler.step()

                total_loss += loss.item()
                progress.set_postfix({"loss": f"{loss.item():.4f}"})

            avg_loss = total_loss / len(self.train_dataloader)
            logger.info(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                save_path = os.path.join(self.output_dir, "best_model.pt")
                torch.save(self.model.state_dict(), save_path)
                logger.info(f"Saved new best model to {save_path}")