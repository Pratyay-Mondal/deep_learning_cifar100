# CIFAR-10 Classification - Correct Implementation

This is a clean, well-structured implementation of CIFAR-10 image classification using PyTorch. It supports both Multi-Layer Perceptron (MLP) and Convolutional Neural Network (CNN) architectures.

## Project Structure

```
.
├── config.py           # Configuration settings
├── models.py           # Model architectures (MLP and CNN)
├── data_loader.py      # Data loading and preprocessing
├── utils.py            # Training and evaluation utilities
├── train.py            # Main training script
├── requirements.txt    # Python dependencies
└── readme.md           # This file
```

## Setup

1. **Install dependencies:**

See *requirements.txt*.

2. **Verify installation:**
```bash
python models.py
python data_loader.py
```

## Usage

### Quick Start

Run training with default settings (CNN model):
```bash
python train.py
```

### Customizing Configuration

Edit `config.py` to change settings.

### Model Architectures

**MLP (Multi-Layer Perceptron):**
- Input: Flattened 32×32×3 images 
- Hidden layers: [512, 256] (configurable)
- BatchNorm + ReLU + Dropout after each hidden layer
- Output: 10 classes

**CNN (Convolutional Neural Network):**
- 2 convolutional blocks (32→64)
- Each block: 2 Conv layers + BatchNorm + ReLU + Dropout + MaxPool 
- Fully connected layers: 512 → 10

### Expected Performance

With default settings:

| Model | Validation Acc | Test Acc | Training Time* |
|-------|---------------|----------|----------------|
| MLP   | ~50-55%       | ~50-55%  | ~5 min         |
| CNN   | ~80-85%       | ~80-85%  | ~15 min        |

*On GPU (NVIDIA RTX 3080)

## Output Files

After training, the following files are created in `./checkpoints/`:

- `best_model_cnn.pth` - Best model checkpoint
- `training_history_cnn.png` - Loss and accuracy curves

## Citation

Dataset: [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
- Learning Multiple Layers of Features from Tiny Images, Alex Krizhevsky, 2009.

## License

This code is for educational purposes.
