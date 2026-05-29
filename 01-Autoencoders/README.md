PyTorch Linear Autoencoder for Representation Learning
This repository contains a clean, modular implementation of a Deep Linear Autoencoder utilizing PyTorch. The network is designed to perform unsupervised representation learning and dimensionality reduction by compressing images into a dense latent space and reconstructing them back into their original form.

The pipeline is fully evaluated on the FashionMNIST dataset, with additional support provided for the classic MNIST handwritten digit dataset.

🚀 Key Features:

1. Modular PyTorch Architecture: Built using nn.Module and sequential API blocks (nn.Sequential).

2. Hardware Acceleration: Automatic GPU acceleration routing (cuda execution) when an NVIDIA GPU is available.

3. Loss Monitoring: Implements Mean Squared Error (MSE) loss metrics optimized with the Adam optimizer.

4. Visual Validation: Scripted evaluation code plotting side-by-side comparative breakdowns of ground truth inputs versus network reconstructions.

📊 Pipeline Architecture:
An autoencoder consists of two complementary components operating together:
  1. Encoder: Downsamples the flattened image input from 784 dimensions (28 * 28 pixels) to a compressed bottleneck representation (Latent Dimension: 64).
  2. Decoder: Upsamples the compressed bottleneck feature representation back to the original dimension of 784 and maps output channels to a range of [0, 1] using a Sigmoid         activation function.

     
Input Frame (784) ──► Linear(256) ──► ReLU ──► Linear(64) ──► [ Latent Bottleneck (z) ]
                                                                      │
Output Frame (784) ◄── Sigmoid ◄── Linear(784) ◄── ReLU ◄── Linear(256) ◄─────┘


🛠️ Code Specifications & Hyperparameters
Input Dimension: 784 (Flattened 28 × 28 single-channel images)

Hidden Dimension: 256

Latent Dimension: 64

Optimizer: Adam (Learning Rate = 0.001)

Loss Criterion: Mean Squared Error (nn.MSELoss)

Batch Size: 128

Training Durations: 5 Epochs

