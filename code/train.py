"""
Main training script for CIFAR-100 classification.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import random

from config import Config
from models import get_model
from data_loader import get_data_loaders, get_cifar100_classes
from utils import (train_one_epoch, evaluate, 
                   plot_training_history, save_checkpoint)


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    """Main training function."""
    
    # Display configuration
    Config.display()
    
    # Set random seed for reproducibility
    set_seed(Config.RANDOM_SEED)
    
    # Create checkpoint directory
    os.makedirs(Config.SAVE_DIR, exist_ok=True)
    
    # Device configuration
    device = torch.device(Config.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load data
    print("\nLoading CIFAR-100 data...")
    train_loader, val_loader, test_loader = get_data_loaders(
        data_dir=Config.DATA_DIR,
        batch_size=Config.BATCH_SIZE,
        val_split=Config.VAL_SPLIT,
        num_workers=Config.NUM_WORKERS,
    )
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    print(f"\nCreating {Config.MODEL_TYPE.upper()} model...")
    if Config.MODEL_TYPE.lower() == 'mlp':
        model = get_model('mlp', 
                         hidden_sizes=Config.MLP_HIDDEN_SIZES,
                         dropout=Config.MLP_DROPOUT,
                         num_classes=Config.NUM_CLASSES)
    else:
        model = get_model('cnn',
                         num_classes=Config.NUM_CLASSES)
    
    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params:,}")
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), 
                            lr=Config.LEARNING_RATE,
                            weight_decay=Config.WEIGHT_DECAY)
    
    # Training loop
    print("\n" + "="*70)
    print("Starting Training")
    print("="*70)
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(1, Config.NUM_EPOCHS + 1):
        # Train for one epoch
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device
        )
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Print progress
        if epoch % Config.PRINT_EVERY == 0 or epoch == 1:
            print(f"Epoch [{epoch:3d}/{Config.NUM_EPOCHS}] | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
              
        # Save best model
        if val_acc > best_val_acc + Config.MIN_DELTA:
            best_val_acc = val_acc
            patience_counter = 0
            
            if Config.SAVE_BEST_ONLY:
                checkpoint_path = os.path.join(Config.SAVE_DIR, 
                                              f'best_model_{Config.MODEL_TYPE}.pth')
                save_checkpoint(model, optimizer, epoch, train_loss, 
                              val_loss, train_acc, val_acc, checkpoint_path)
        else:
            patience_counter += 1
        
        # Early stopping
        if Config.USE_EARLY_STOPPING and patience_counter >= Config.PATIENCE:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            print(f"Best validation accuracy: {best_val_acc:.2f}%")
            break
    
    print("\n" + "="*70)
    print("Training Completed")
    print("="*70)
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    
    # Plot training history
    print("\nGenerating training history plots...")
    plot_training_history(train_losses, val_losses, train_accs, val_accs,
                         save_path=os.path.join(Config.SAVE_DIR, 
                                               f'training_history_{Config.MODEL_TYPE}.png'))
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.2f}%")
     
    print("\n" + "="*70)
    print("All Done!")
    print("="*70)


if __name__ == "__main__":
    main()
