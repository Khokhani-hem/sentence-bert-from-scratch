import argparse
import yaml
import torch
import numpy as np
from pathlib import Path

from src.utils.logger import get_logger
from src.data.preprocessing import prepare_stsb_dataset
from src.data.loader import get_dataloader
from src.models.siamese import SiameseNetwork
from src.training.evaluator import STSEvaluator
from src.utils.visualization import EmbeddingVisualizer

def main():
    parser = argparse.ArgumentParser(description="Evaluate Trained Siamese Network")
    parser.add_argument("--model_config", type=str, default="configs/model.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pt")
    args = parser.parse_args()

    logger = get_logger(__name__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with open(args.model_config, "r") as f:
        model_config = yaml.safe_load(f)

    test_ds = prepare_stsb_dataset(split="validation")

    vis_subset = test_ds.select(range(300)) 
    test_dl = get_dataloader(test_ds, batch_size=32, is_regression=True, shuffle=False)


    logger.info(f"Loading model from {args.checkpoint}...")
    model = SiameseNetwork(
        model_name=model_config["model"]["backbone"],
        max_seq_length=model_config["model"]["max_seq_length"],
        pooling_strategy=model_config["model"]["pooling_strategy"]
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)

    logger.info("Starting Quantitative Evaluation on STS-B Test Set...")
    evaluator = STSEvaluator(dataloader=test_dl, device=device)
    results = evaluator(model)
    
    print("\n" + "="*40)
    print("FINAL TEST RESULTS")
    print("="*40)
    print(f"Pooling Strategy : {model_config['model']['pooling_strategy'].upper()}")
    print(f"Pearson (r)      : {results['pearson']:.4f}")
    print(f"Spearman (rho)   : {results['spearman']:.4f}")
    print("="*40 + "\n")

    logger.info("Generating UMAP Visualization...")

    texts_a = vis_subset["text_a"]
    labels = np.array(vis_subset["label"])
    
    embeddings = model.encode(texts_a, device=device, batch_size=32).numpy()
    
    visualizer = EmbeddingVisualizer(method="umap")
    plot_path = Path(args.checkpoint).parent / "umap_embeddings.html"
    
    visualizer.plot_interactive(
        embeddings=embeddings,
        texts=texts_a,
        labels=labels,
        title=f"STS-B Embeddings - {model_config['model']['pooling_strategy'].upper()} Pooling",
        save_path=str(plot_path)
    )
    logger.info(f"Evaluation complete. Visualization saved to {plot_path}")

if __name__ == "__main__":
    main()
