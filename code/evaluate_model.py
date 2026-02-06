
"""
Evaluation script to load the best trained model and evaluate on CIFAR-100 test set.
 reproducibility of the reported results.
"""

import torch
import torch.nn as nn
import os
from config import Config
from models import get_model
from data_loader import get_data_loaders
from utils import evaluate

def load_and_evaluate():
    # 1. Device Configuration
    device = torch.device(Config.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"\nEvaluating on device: {device}")

    # 2. Load Data (CIFAR-100 Test Set)
    print("Loading test data...")
    # We only need the test_loader, but the function returns all three
    _, _, test_loader = get_data_loaders(
        data_dir=Config.DATA_DIR,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS
    )
    print(f"Test samples: {len(test_loader.dataset)}")

    # 3. Initialize Model Architecture
    # should match the architecture used during training (CNN, 100 classes)
    print(f"Initializing {Config.MODEL_TYPE.upper()} model...")
    model = get_model(Config.MODEL_TYPE, num_classes=100)
    model = model.to(device)

    # 4. Load the Best Checkpoint
    checkpoint_path = os.path.join(Config.SAVE_DIR, f'best_model_{Config.MODEL_TYPE}.pth')
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Did you run train.py?")
        
    print(f"Loading weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle loading: check if checkpoint is full dict or just weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch']
        print(f"Loaded checkpoint from Epoch {epoch}")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded model weights (no epoch info)")

    # 5. Run Evaluation
    print("Running evaluation on Test Set...")
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    # 6. Report Results
    print("\n" + "="*40)
    print(f"FINAL RESULT: Task 2 (CIFAR-100)")
    print("="*40)
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Test Loss:     {test_loss:.4f}")
    print("="*40)

if __name__ == "__main__":
    load_and_evaluate()

