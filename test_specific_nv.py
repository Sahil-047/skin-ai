"""
Test specific image that's showing acne prediction
"""

import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path
from PIL import Image
import torchvision.transforms as T

def test_specific_image():
    """Test a specific image that's showing acne prediction."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 9)
    
    model_path = Path("training/best_model.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Class names
    CLASS_NAMES = {
        0: 'akiec', 1: 'bcc', 2: 'bkl', 
        3: 'df', 4: 'mel', 5: 'nv', 6: 'vasc',
        7: 'non_skin', 8: 'acne'
    }
    
    # Test a few random nv images
    test_images = [
        "ISIC_0024698.jpg",
        "ISIC_0024693.jpg", 
        "ISIC_0032212.jpg",
        "ISIC_0025775.jpg",
        "ISIC_0029828.jpg"
    ]
    
    print("Testing specific melanocytic nevi images:")
    print("="*50)
    
    for img_name in test_images:
        img_path = Path("data/images") / img_name
        
        if img_path.exists():
            try:
                # Load and preprocess image (same as backend)
                image = Image.open(img_path).convert("RGB")
                
                transform = T.Compose([
                    T.Resize((224, 224)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
                
                input_tensor = transform(image).unsqueeze(0).to(device)
                
                # Get prediction
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    predicted_class = outputs.argmax(dim=1).item()
                    confidence = probabilities[0][predicted_class].item()
                
                # Get top 3 predictions
                top3_probs, top3_indices = torch.topk(probabilities[0], 3)
                
                print(f"Image: {img_name}")
                print(f"Predicted: {CLASS_NAMES[predicted_class]} ({confidence:.3f})")
                print(f"Top 3: {[(CLASS_NAMES[idx.item()], f'{prob.item():.3f}') for idx, prob in zip(top3_indices, top3_probs)]}")
                
                if predicted_class == 8:  # acne
                    print("WARNING: Predicted as ACNE!")
                elif predicted_class == 5:  # nv
                    print("CORRECT: Predicted as melanocytic nevi")
                else:
                    print(f"Predicted as: {CLASS_NAMES[predicted_class]}")
                
                print("-" * 30)
                
            except Exception as e:
                print(f"Error processing {img_name}: {e}")
        else:
            print(f"Image not found: {img_path}")
    
    print("\nIf you're seeing acne predictions, please:")
    print("1. Share the specific image filename")
    print("2. Check if you're using the updated backend")
    print("3. Verify the image is in data/images/ directory")

if __name__ == "__main__":
    test_specific_image()
