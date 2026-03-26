# CNN from Scratch to Transfer Learning

A hands-on Jupyter notebook covering the fundamentals of **Convolutional Neural Networks (CNNs)**, from low-level NumPy implementations to fine-tuning a pretrained ResNet-18 model with experiment tracking via Weights & Biases.

---

## What's Inside

### Part 1 — CNN Building Blocks (NumPy)
Implementing the core CNN operations from scratch, without any deep learning framework, to build intuition for how they work:

- **Zero Padding** — manually pad images before applying filters
- **Convolution** — sliding a kernel over an image to compute feature maps
  - *Low-pass filters*: Mean (3×3, 9×9) and Gaussian — blur/smooth images
  - *High-pass filters*: Sobel X & Y — detect horizontal and vertical edges
- **Pooling** — reduce spatial dimensions using Max or Average pooling

All operations are tested on both grayscale (`cameraman.jpg`) and RGB (`orange.jpg`) images.

---

### Part 2 — CNNs in PyTorch
Moving to PyTorch to build and train real networks on the [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) (37 cat/dog breeds):

**Simple CNN from scratch**
- 2 convolutional layers + 3 fully connected layers
- Trained with SGD + CrossEntropyLoss

**Transfer Learning with ResNet-18**
- Load ResNet-18 pretrained on ImageNet
- Freeze all backbone weights; replace only the final classification head
- Fine-tune on the Pet dataset

---

### Part 3 — Experiment Tracking with W&B
Using [Weights & Biases](https://wandb.ai) to systematically compare training runs:

- Test multiple learning rates: `0.01`, `0.001`, `0.0001`
- Compare LR schedulers: `StepLR` vs `ExponentialLR`
- Track train/test loss and accuracy per epoch across all runs

- You can see the wandb reports here:
- https://api.wandb.ai/links/dragostrandafir443-babes/j2ij00a6 - transfer-learning-lr in the notebook
- https://api.wandb.ai/links/dragostrandafir443-babes/gtnn58yn - transfer-learning-sweep1 in the notebook
- https://api.wandb.ai/links/dragostrandafir443-babes/k5ghfbs4 - transfer-learning-sweep2 in the notebook

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python / NumPy | From-scratch convolution & pooling |
| PyTorch + torchvision | Model definition, training, dataset loading |
| Weights & Biases (wandb) | Experiment tracking and visualization |
| OpenCV / PIL / Matplotlib | Image loading and visualization |

---

## How to Run

1. Open the notebook in [Google Colab](https://colab.research.google.com/) or locally with Jupyter
2. Enable GPU: *Runtime → Change runtime type → T4 GPU* (Colab)
3. Install dependencies if needed:
   ```bash
   pip install torch torchvision wandb opencv-python
   ```
4. Run cells top to bottom — datasets are downloaded automatically

---

## Key Concepts Covered

- Convolution hyperparameters: filter size, padding, stride
- Feature maps and receptive fields
- Max vs Average pooling
- The PyTorch `nn.Module` pattern (constructor + `forward()`)
- Transfer learning: when and why to freeze layers
- Learning rate scheduling and its effect on convergence
