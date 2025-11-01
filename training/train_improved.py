"""
Improved Training Script for Skin Disease Classification

Key improvements:
1. Weighted loss to handle class imbalance
2. Better data augmentation (more aggressive)
3. Learning rate scheduling
4. Gradient clipping
5. Better model architecture (ResNet50 instead of ResNet18)
6. Early stopping
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import models
import pandas as pd
from dataset import SkinDataset
from transformer import get_train_transforms, get_val_transforms
from pathlib import Path


def create_model(num_classes=15, pretrained=True, arch='resnet50'):
    """Create a model for skin disease classification."""
    if arch == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
    else:
        model = models.resnet18(pretrained=pretrained)
    
    # Replace final layer for our classes
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    return model


def calculate_class_weights(csv_path):
    """Calculate class weights for handling imbalanced dataset."""
    df = pd.read_csv(csv_path)
    class_counts = df['dx'].value_counts()
    
    # Calculate inverse frequency weights
    total_samples = len(df)
    num_classes = len(class_counts)
    class_weights = {}
    
    for dx in df['dx'].unique():
        class_weights[dx] = total_samples / (num_classes * class_counts[dx])
    
    # Create weight tensor aligned with label_id (one weight per class)
    df_sorted = df.sort_values('label_id')
    unique_labels = df_sorted[['dx', 'label_id']].drop_duplicates()
    
    weight_dict = {}
    weights = [0.0] * 15  # Initialize with 15 classes
    
    for _, row in unique_labels.iterrows():
        dx = row['dx']
        label_id = row['label_id']
        weights[label_id] = class_weights[dx]
        weight_dict[dx] = class_weights[dx]
    
    return torch.FloatTensor(weights), weight_dict


def train_one_epoch(model, dataloader, criterion, optimizer, device, grad_clip=None):
    """Train for one epoch with gradient clipping."""
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
        
        # Gradient clipping
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if (batch_idx + 1) % 20 == 0:
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
    print("IMPROVED SKIN DISEASE CLASSIFICATION TRAINING")
    print("="*70)
    
    # Configuration (paths relative to repo root, robust to CWD)
    repo_root = Path(__file__).resolve().parents[1]
    CSV_PATH = str((repo_root / "data" / "final_expanded_dataset.csv").resolve())
    IMG_DIR = str((repo_root / "data" / "images").resolve())
    NON_SKIN_DIR = str((repo_root / "data" / "non_skin").resolve())
    DERMNET_DIR = str((repo_root / "data" / "dermnet_images").resolve())
    
    # Training parameters
    BATCH_SIZE = 32
    NUM_EPOCHS = 20  # More epochs
    LEARNING_RATE = 0.0001  # Lower initial LR
    TRAIN_SPLIT = 0.8
    GRAD_CLIP = 1.0  # Gradient clipping
    MODEL_ARCH = 'resnet50'  # Better model
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"Model: {MODEL_ARCH}")
    
    # Calculate class weights for imbalanced dataset
    print("\nCalculating class weights for imbalanced dataset...")
    class_weights, weight_dict = calculate_class_weights(CSV_PATH)
    print("Class weights (sample):")
    for i, (dx, weight) in enumerate(list(weight_dict.items())[:5]):
        print(f"  {dx}: {weight:.2f}")
    print(f"  ... ({len(weight_dict)} total classes)")
    
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
    print("\nCreating model (ResNet50, 15 classes)...")
    model = create_model(num_classes=15, pretrained=True, arch=MODEL_ARCH)
    model = model.to(device)
    
    # Weighted loss for imbalanced dataset
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    # AdamW optimizer with better defaults
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    
    # Learning rate scheduler - reduce LR when plateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    print(f"  Optimizer: AdamW (lr={LEARNING_RATE})")
    print(f"  Loss function: Weighted CrossEntropyLoss")
    print(f"  Scheduler: ReduceLROnPlateau")
    
    # Training loop
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)
    
    best_val_acc = 0.0
    patience_counter = 0
    max_patience = 5  # Early stopping
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}]")
        print("-" * 70)
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, grad_clip=GRAD_CLIP
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print results
        print(f"\nEpoch {epoch+1} Results:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
            print(f"  Saved best model (val_acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{max_patience})")
        
        # Early stopping
        if patience_counter >= max_patience:
            print("\nEarly stopping triggered!")
            break
    
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
        import traceback
        traceback.print_exc()

