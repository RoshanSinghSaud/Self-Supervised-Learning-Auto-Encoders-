# Self-Supervised-Learning-Auto-Encoders-

🧠 What is Self-Supervised Learning?
Self-Supervised Learning (SSL) is a subset of machine learning where the model generates its own "labels" from the raw data. It works by hiding part of the input (or transforming it) and tasking the model with predicting the missing information.
Core Philosophy: "The data is the supervision." By solving "pretext tasks," models learn deep structural features that often outperform supervised models in downstream tasks like detection or segmentation.

🛠️ Project Evolution
01. Basic Autoencoders (AE):

  Target: MNIST (28×28)
  Concept: A bottleneck architecture (Encoder-Decoder) that learns a compressed latent representation to reconstruct the input.
  Key Learning: Understanding dimensionality reduction and the "Information Bottleneck."

02. Masked Autoencoders (MAE):

  Target: CIFAR-10 (32×32×3)
  Concept: Based on the Vision Transformer (ViT) architecture. We mask 75% of image patches and challenge the Transformer to "imagine" the missing pixels.
  Key Learning: Implementation of random masking logic, patch-based reconstruction, and scaling Transformers for generative tasks.
