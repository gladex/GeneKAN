# [Review]
A Gene Regulatory Network Inference Method Based on Multiscale KAN-Enhanced Graph Convolutional Kernels

---
##### Overview

we propose GeneKAN, a new method that integrates a kernel-based graph convolutional network, termed Graph Convolution with Kernel Methods (GCKM), to infer GRNs from single-cell transcriptomic data. By leveraging kernel methods, our model generated enriched feature representations based on pairwise gene similarities. By stacking multiple GCKM layers with varying kernel widths, which modulate the receptive field of each node, the model captures complex gene interactions and nonlinear dependencies in a high-dimensional feature space, effectively encoding global structural dependencies.

----
## Environment Setup

This codebase runs on:

- numpy==1.19.3
- pandas==1.2.4
- scikit_learn==1.0.2
- scipy==1.7.3

### Installation

Install dependencies using the trimmed core requirement list:

```bash
pip install -r requirements.txt
```
