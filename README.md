# GeneKAN

[Review] Integrative Graph Deep Learning with Directional Decoding for Robust Gene Regulatory Network Inference from Single-Cell Transcriptomics
---

## Features

- **Encoder** with dual TF-target channels.
- **Asymmetric Relation Decoder** for TF → Target relation prediction.
- **Node Autoencoder** for feature reconstruction regularization.
- Supports AUC, AUPR evaluation and TF/Target embedding export.

---

## Environment Setup

This codebase runs on:

- Python 3.9.20
- PyTorch 2.3.1 (CUDA 12.1)

### Installation

Install dependencies using the trimmed core requirement list:

```bash
pip install -r requirements.txt
```

## Training the Model

Run the training script with desired parameters:

```bash
python main.py --data hHEP --num 500 --net Specific --lr 0.003 --epochs 20 --normalize True --device cuda
```

### Common Arguments

| Argument         | Description                                 |
|------------------|---------------------------------------------|
| `--data`         | Dataset name (e.g. `hHEP`)                  |
| `--net`          | Network type: `Specific`, `STRING`, etc.   |
| `--num`          | Number of genes (e.g. 500, 1000)            |
| `--normalize`    | Whether to apply feature normalization      |
| `--flag`         | Enable one-hot causal label format          |
| `--loop`         | Add self-loops to adjacency matrix          |
| `--lr`           | Learning rate                               |
| `--epochs`       | Number of training epochs                   |
| `--device`       | `cpu` or `cuda`                             |

---

## Evaluation Metrics

- **AUROC**: Area Under ROC Curve  
- **AUPR**: Area Under Precision-Recall Curve  
- **AUPR_norm**: AUPR normalized by positive rate

Results are printed each epoch and final metrics are shown on the test set.
