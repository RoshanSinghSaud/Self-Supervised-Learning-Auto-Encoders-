# Masked Autoencoders (MAE) Implementation from Scratch

This repository contains a clean, modular, from-scratch implementation of a Masked Autoencoder (MAE) in PyTorch. The architecture is designed and optimized specifically for self-supervised pre-training on the CIFAR-10 dataset using an asymmetric Vision Transformer (ViT) pipeline.

MAE is a powerful self-supervised learning paradigm where a large portion of image patches (e.g., 75%) are masked out, and the model is tasked with reconstructing the missing pixels. This implementation provides the full pipeline from raw patch embedding, random token masking, asymmetric transformer encoding/decoding, down to patch reconstruction and visualization.

## Key Features
- **Patch Embedding Head**: Converts continuous images into discrete structured visual tokens using a 2D convolution sequence.
- **Asymmetric Encoder-Decoder Pipeline**: A heavy Transformer Encoder processes only the unmasked visual tokens, while a lightweight Transformer Decoder reconstructs the entire sequence.
- **Random Vectorized Masking**: Implements an optimized masking function using argsort and gather, avoiding iterative loops.
- **Selective Loss Accumulation**: The Mean Squared Error (MSE) reconstruction loss is computed strictly over the masked visual patches.
- **Verification Overlay**: A complete post-training visualization script to display the Original, Masked, and Reconstructed images side-by-side.

## Model Architecture & Hyperparameters
The model configurations are calibrated for optimal feature learning on CIFAR-10 images (32x32 pixels):

- **Image Size**: 32 x 32
- **Patch Size**: 4 x 4 (resulting in 64 total patches per image)
- **Input Channels**: 3 (RGB)
- **Embedding Dimension**: 256
- **Attention Heads**: 16
- **MLP Nodes (Hidden Dim)**: 512
- **Encoder Depth**: 4 Transformer blocks
- **Decoder Depth**: 2 Transformer blocks
- **Default Masking Ratio**: 75% (0.75)

## Installation & Requirements
To run this project, ensure you have Python 3.8+ installed along with the following standard deep learning libraries:

```bash
pip install torch torchvision matplotlib numpy
