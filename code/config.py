"""
Configuration file for CIFAR-100 training.
"""

class Config:
    """Configuration parameters for training."""
    
    # Model settings
    MODEL_TYPE = 'cnn'  # 'mlp' or 'cnn'
    
    # MLP specific settings
    MLP_HIDDEN_SIZES = [512, 256]
    MLP_DROPOUT = 0.3
    
    # Data settings
    DATA_DIR = './data'
    BATCH_SIZE = 128
    VAL_SPLIT = 0.1  # Fraction of training data for validation
    NUM_WORKERS = 2
    
    # Training settings
    NUM_EPOCHS = 20
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 5e-4
    
    # Early stopping
    USE_EARLY_STOPPING = True
    PATIENCE = 10
    MIN_DELTA = 0.001  # Minimum change to qualify as improvement
    
    # Device
    DEVICE = 'cpu'  # 'cuda' or 'cpu', will auto-detect if cuda available
    
    # Checkpointing
    SAVE_DIR = '../checkpoints'
    SAVE_BEST_ONLY = True
    
    # Logging
    PRINT_EVERY = 5  # Print training stats every N epochs
    
    # Reproducibility
    RANDOM_SEED = 42
    
    # Number of classes
    NUM_CLASSES = 100
    
    @classmethod
    def display(cls):
        """Display all configuration parameters."""
        print("=" * 70)
        print("Configuration Settings")
        print("=" * 70)
        for attr in dir(cls):
            if not attr.startswith('_') and attr.isupper():
                print(f"{attr:.<30} {getattr(cls, attr)}")
        print("=" * 70)


if __name__ == "__main__":
    Config.display()
