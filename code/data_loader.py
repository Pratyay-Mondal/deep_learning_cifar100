"""
Data loading and preprocessing for CIFAR-100.
"""
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


def get_data_loaders(data_dir='./data', batch_size=128, val_split=0.1, 
                     num_workers=2):
    """
    Create train, validation, and test data loaders for CIFAR-100.
    
    Args:
        data_dir: Directory to store/load CIFAR-100 data
        batch_size: Batch size for training
        val_split: Fraction of training data to use for validation
        num_workers: Number of workers for data loading
    
    Returns:
        train_loader, val_loader, test_loader
    """
    
    # CIFAR-10 normalization constants (mean and std per channel)
    # mean = [0.4914, 0.4822, 0.4465]
    # std = [0.2023, 0.1994, 0.2010]
    
    # CIFAR-100 normalization constants (mean and std per channel)
    mean = [0.5071, 0.4867, 0.4408]
    std = [0.2675, 0.2565, 0.2761]


    # 1. Stronger Training Transform (data Augmentation)
    # this prevent the 30% Gap by constantly changing the input
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),       # randomly Shift image slightly
        transforms.RandomHorizontalFlip(),          # Mirror image (like truck faces)
        transforms.RandomRotation(15),              # Rotate between -15 and +15 degrees
        transforms.ColorJitter(brightness=0.2, contrast=0.2), # Random lighting changes
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # 2.  Test/Val Transform (No Augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Data transforms
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
       
    # Load full training dataset
    full_train_dataset = datasets.CIFAR100(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform
    )
    
    # Split into train and validation
    val_size = int(len(full_train_dataset) * val_split)
    train_size = len(full_train_dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(2025)  # For reproducibility
    )
      
    # Load test dataset
    test_dataset = datasets.CIFAR100(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def get_cifar100_classes():
    """Return the list of CIFAR-100 class names."""
    return ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
        'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
        'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
        'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
        'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
        'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 
        'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 
        'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 
        'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 
        'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 
        'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 
        'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 
        'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 
        'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']


if __name__ == "__main__":
    # Test the data loader
    print("Loading CIFAR-100 data...")
    train_loader, val_loader, test_loader = get_data_loaders(batch_size=64)
    
    print(f"\nDataset sizes:")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    print(f"\nNumber of batches:")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Check a batch
    images, labels = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"Images: {images.shape}")
    print(f"Labels: {labels.shape}")
    print(f"Image value range: [{images.min():.3f}, {images.max():.3f}]")
    
    print(f"\nClasses: {get_cifar100_classes()}")
