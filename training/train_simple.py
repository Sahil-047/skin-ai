"""
Simple Training Example for Skin Disease Classification

This script demonstrates a basic training loop using the SkinDataset.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import models
from dataset import SkinDataset
from transformer import get_train_transforms, get_val_transforms
from pathlib import Path


def create_model(num_classes=10, pretrained=True):
    """Create a ResNet18 model for skin disease classification."""
    model = models.resnet18(pretrained=pretrained)
    
    # Replace final layer for our classes
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    return model


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels, metadata) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch [{batch_idx+1}/{len(dataloader)}] "
                  f"Loss: {loss.item():.4f} "
                  f"Acc: {100.*correct/total:.2f}%")
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels, metadata in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    val_loss = running_loss / len(dataloader)
    val_acc = 100. * correct / total
    
    return val_loss, val_acc


def main():
    """Main training function."""
    
    print("\n" + "="*70)
    print("SKIN DISEASE CLASSIFICATION TRAINING")
    print("="*70)
    
    # Configuration (paths relative to repo root, robust to CWD)
    repo_root = Path(__file__).resolve().parents[1]
    CSV_PATH = str((repo_root / "data" / "expanded_with_athletes_foot.csv").resolve())
    IMG_DIR = str((repo_root / "data" / "images").resolve())
    NON_SKIN_DIR = str((repo_root / "data" / "non_skin").resolve())
    DERMNET_DIR = str((repo_root / "data" / "dermnet_images").resolve())
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.001
    TRAIN_SPLIT = 0.8
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    # Create datasets
    print("\nLoading datasets...")
    train_dataset = SkinDataset(
        csv_path=CSV_PATH,
        img_dir=IMG_DIR,
        transform=get_train_transforms(),
        non_skin_dir=NON_SKIN_DIR,
        dermnet_dir=DERMNET_DIR
    )
    
    val_dataset = SkinDataset(
        csv_path=CSV_PATH,
        img_dir=IMG_DIR,
        transform=get_val_transforms(),
        non_skin_dir=NON_SKIN_DIR,
        dermnet_dir=DERMNET_DIR
    )
    
    # Split data
    total_size = len(train_dataset)
    train_size = int(TRAIN_SPLIT * total_size)
    val_size = total_size - train_size
    
    train_indices, val_indices = random_split(
        range(total_size),
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_subset = torch.utils.data.Subset(train_dataset, train_indices.indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices.indices)
    
    print(f"  Training samples: {train_size}")
    print(f"  Validation samples: {val_size}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Create model
    print("\nCreating model (ResNet18, 10 classes)...")
    model = create_model(num_classes=10, pretrained=True)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"  Optimizer: Adam (lr={LEARNING_RATE})")
    print(f"  Loss function: CrossEntropyLoss")
    
    # Training loop
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)
    
    best_val_acc = 0.0
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}]")
        print("-" * 70)
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Print results
        print(f"\nEpoch {epoch+1} Results:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print(f"  Saved best model (val_acc: {val_acc:.2f}%)")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: best_model.pth")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure:")
        print("  1. You've run metadata.py to create metadata_final.csv")
        print("  2. Images are in data/images/ folder")
        print("  3. PyTorch is installed: pip install torch torchvision")
