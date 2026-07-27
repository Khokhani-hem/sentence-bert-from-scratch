# sentence-bert-from-scratch
A research-grade PyTorch implementation of Sentence-BERT built from scratch, featuring Siamese BERT, multiple pooling strategies (Mean, CLS, Max), metric learning objectives (Cosine, MNRL), STS evaluation, and reproducible experiments using uv.

## Overview

This project reproduces the core architecture proposed in the Sentence-BERT (SBERT) paper while emphasizing software engineering, modular design, and reproducible experimentation.

Instead of using high-level abstraction libraries, every major component—including the Siamese architecture, pooling layers, metric learning objectives, training pipeline, evaluation pipeline, and visualization tools—is implemented manually.

The primary objective is to understand how sentence embedding models are built internally while providing a flexible research framework for experimenting with different pooling strategies and loss functions.

---

## Motivation

Standard BERT works as a **cross-encoder**, requiring both sentences to be processed together.

For semantic search, comparing one query against thousands of documents requires thousands of forward passes.

Sentence-BERT solves this by learning independent sentence embeddings using a Siamese network, allowing semantic similarity to be computed with simple vector operations.

Benefits include:

- Fast semantic search
- Efficient information retrieval
- Scalable sentence embedding generation
- Vector database compatibility (FAISS, Milvus, Pinecone, etc.)

---

# Features

- Custom SBERT implementation from scratch
- Siamese BERT architecture
- Shared-weight encoder
- Multiple pooling strategies
    - Mean Pooling
    - CLS Pooling
    - Max Pooling
- Multiple metric learning objectives
    - Cosine Similarity Loss
    - Multiple Negatives Ranking Loss (MNRL)
- STS Benchmark evaluation
- Natural Language Inference (SNLI + MultiNLI) training
- Pearson & Spearman correlation evaluation
- UMAP embedding visualization
- YAML-based experiment configuration
- Fully reproducible training pipeline

---

# Architecture

```
                Sentence A
                     │
              Bert Encoder
                     │
             Pooling Layer
                     │
              Sentence Vector
                     │
                     │

                Sentence B
                     │
              Bert Encoder
      (shared parameters)
                     │
             Pooling Layer
                     │
              Sentence Vector

                     │
                     ▼

          Metric Learning Loss
        (Cosine / MNRL / Others)
```

The implementation separates every major component into independent modules, making it easy to extend or replace any part of the architecture.

---

# Pooling Strategies

Implemented pooling methods:

### Mean Pooling

The average of all token embeddings while masking padding tokens.

Recommended for semantic similarity.

---

### CLS Pooling

Uses the hidden state of the `[CLS]` token.

Simple but generally weaker than mean pooling.

---

### Max Pooling

Applies element-wise max over token embeddings after masking padding values.

Useful for experimentation.

---

# Loss Functions

## Cosine Similarity Loss

Used for regression-based Semantic Textual Similarity datasets.

Dataset:

- STS Benchmark

---

## Multiple Negatives Ranking Loss (MNRL)

Contrastive learning objective where all other examples in a batch become implicit negatives.

Dataset:

- SNLI
- MultiNLI

---

# Datasets

| Dataset | Purpose |
|----------|----------|
| STS Benchmark | Semantic Textual Similarity |
| SNLI | Natural Language Inference |
| MultiNLI | Metric Learning |

---

# Evaluation Metrics

The implementation supports:

- Pearson Correlation
- Spearman Correlation

These metrics evaluate how well embedding similarities correlate with human-annotated semantic similarity scores.

---

# Experimental Results

| Pooling | Loss | Dataset | Pearson | Spearman |
|----------|------|----------|---------|-----------|
| Mean | Cosine | STS-B | **0.8765** | **0.8764** |
| CLS | Cosine | STS-B | 0.8578 | 0.8614 |
| Max | Cosine | STS-B | 0.7459 | 0.7400 |
| Mean | MNRL | NLI | 0.8379 | 0.8361 |

Mean pooling consistently provides the strongest performance for semantic similarity tasks.

---

# Project Structure

```
Sentence-BERT/
│
├── configs/
│   ├── model.yaml
│   └── train.yaml
│
├── data/
│
├── outputs/
│   ├── checkpoints/
│   ├── logs/
│   └── visualizations/
│
├── src/
│   ├── datasets/
│   ├── losses/
│   ├── models/
│   │   ├── bert_encoder.py
│   │   ├── pooling.py
│   │   └── siamese_network.py
│   │
│   ├── trainer/
│   ├── evaluator/
│   ├── utils/
│   └── visualization.py
│
├── train.py
├── evaluate.py
├── pyproject.toml
├── uv.lock
└── README.md
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/Khokhani-hem/sentence-bert-from-scratch.git

cd sentence-bert-from-scratch
```

Install dependencies using **uv**

```bash
uv sync
```

---

# Training

```bash
uv run train.py
```

---

# Evaluation

```bash
uv run evaluate.py \
--checkpoint outputs/checkpoints/cosine_mean/best_model.pt
```

---

# Visualization

Generate a 2D UMAP projection of sentence embeddings.

The visualization helps inspect clustering quality and embedding space structure.

---

# Configuration

Experiments are controlled entirely through YAML files.

Example:

```yaml
pooling_strategy: mean

loss_type: cosine

backbone: bert-base-uncased

batch_size: 16

learning_rate: 2e-5

epochs: 4
```

Changing a single configuration file is sufficient to reproduce a different experiment.

---

# Software Stack

- Python 3.12+
- PyTorch
- Hugging Face Transformers
- Datasets
- SciPy
- UMAP
- NumPy

Dependency management is handled with **uv**.

---

# Reproducibility

This repository was designed with reproducible research in mind.

Features include:

- Fixed random seeds
- Deterministic CUDA execution
- YAML-driven experiments
- Modular architecture
- Version-controlled dependencies
- uv package management

---

# Future Work

Potential extensions include:

- Triplet Loss
- Contrastive Loss
- ColBERT integration
- MS MARCO retrieval evaluation
- FAISS indexing
- Dense Passage Retrieval
- Cross-dataset transfer learning

---

# References

- Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.
- Devlin, J., et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers.
- Vaswani, A., et al. (2017). Attention Is All You Need.
- Wang, A., et al. (2018). GLUE Benchmark.
- McInnes, L., et al. UMAP: Uniform Manifold Approximation and Projection.

---

## Citation

If this repository contributes to your research, please consider citing the original Sentence-BERT paper.

```bibtex
@inproceedings{reimers2019sentencebert,
  title={Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks},
  author={Reimers, Nils and Gurevych, Iryna},
  booktitle={EMNLP},
  year={2019}
}
```

---

## License

This project is intended for educational and research purposes.
