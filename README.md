# grokking
reproduce grokking—a delayed generalization transition—in a transformer trained on modular addition

**The research results are documented in report2.pdf**

---

## Overview
Neural networks often exhibit emergent abilities, particularly during the pretraining stage of Large Language Models (LLMs) and Transformers. This project explores "grokking"—a surprising phenomenon where a neural network suddenly achieves perfect generalization to unseen data after an extended period of training—as a key to understanding these emergent capabilities.

In this project, I have successfully:
- Reproduced grokking: Observed the delayed generalization transition in a Transformer model trained on modular addition tasks.
- Applied mechanistic interpretability: Identified and analyzed the specific algorithms learned by the model during the training process.

Following the methodology of "Progress Measures for Grokking via Mechanistic Interpretability" (ICLR 2023), I implemented a single-layer Transformer and trained it extensively to observe the grokking transition. To decode the network's internal logic, I conducted in-depth analyses using Fourier analysis, Attention/MLP visualizations, and low-rank structure identification.

---
## Part 1. Reproduce grokking (training + plots)
Goal: show memorization (high train accuracy, low test accuracy) followed by a sudden improvement in test accuracy after many epochs.

## Part 2. Reverse engineer the learned algorithm
Goal: identify which Fourier modes dominate the computation and connect them to periodic structure in attention/MLP activations and output weights.

## Part 3. Training dynamics and progress measures
Goal: identify three phases of training and connect them to mechanistic progress measures.
