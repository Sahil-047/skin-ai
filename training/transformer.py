import torchvision.transforms as T

def get_train_transforms(img_size=224):
    """Enhanced data augmentation for better generalization."""
    return T.Compose([
        T.Resize((img_size + 32, img_size + 32)),  # Slightly larger for cropping
        T.RandomCrop(img_size, padding=4),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.3),  # Additional augmentation
        T.RandomRotation(degrees=20),  # More rotation
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),  # More aggressive
        T.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Translation
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.RandomErasing(p=0.2),  # Random erasing augmentation
    ])

def get_val_transforms(img_size=224):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
