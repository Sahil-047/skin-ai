"""
Test the specific type of melanocytic nevus image described
"""

import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path
import pandas as pd
from PIL import Image
import torchvision.transforms as T

def test_dark_nevi_images():
    """Test dark, irregular melanocytic nevi images."""
    
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
    
    # Load metadata and find dark/irregular nv images
    df = pd.read_csv('data/expanded_with_acne.csv')
    nv_images = df[df['dx'] == 'nv']
    
    print("Testing dark/irregular melanocytic nevi images:")
    print("="*60)
    
    # Test several nv images to see if any are misclassified as acne
    acne_misclassifications = 0
    total_tested = 0
    
    for idx, row in nv_images.head(20).iterrows():  # Test more images
        img_name = f"{row['image']}.jpg"
        img_path = Path("data/images") / img_name
        
        if img_path.exists():
            try:
                # Load and preprocess image
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
                
                total_tested += 1
                
                if predicted_class == 8:  # acne
                    acne_misclassifications += 1
                    print(f"❌ MISCLASSIFIED: {img_name}")
                    print(f"   Predicted: {CLASS_NAMES[predicted_class]} ({confidence:.3f})")
                    print(f"   Top 3: {[(CLASS_NAMES[idx.item()], f'{prob.item():.3f}') for idx, prob in zip(top3_indices, top3_probs)]}")
                    print("-" * 40)
                elif predicted_class != 5:  # not nv
                    print(f"⚠️  WRONG CLASS: {img_name}")
                    print(f"   Predicted: {CLASS_NAMES[predicted_class]} ({confidence:.3f})")
                    print(f"   Top 3: {[(CLASS_NAMES[idx.item()], f'{prob.item():.3f}') for idx, prob in zip(top3_indices, top3_probs)]}")
                    print("-" * 40)
                
            except Exception as e:
                print(f"Error processing {img_name}: {e}")
    
    print(f"\nSummary:")
    print(f"Total nv images tested: {total_tested}")
    print(f"Misclassified as acne: {acne_misclassifications}")
    print(f"Accuracy: {(total_tested - acne_misclassifications)/total_tested*100:.1f}%")
    
    if acne_misclassifications > 0:
        print(f"\n🚨 PROBLEM FOUND!")
        print(f"The model is misclassifying {acne_misclassifications} melanocytic nevi as acne.")
        print("This is a serious issue for medical diagnosis.")
        print("\nPossible causes:")
        print("1. Insufficient acne training data (only 34 images)")
        print("2. Model overfitting to acne features")
        print("3. Class imbalance causing confusion")
        print("\nSolutions:")
        print("1. Add more acne training data")
        print("2. Use class weights during training")
        print("3. Remove acne class temporarily")
    else:
        print("✅ No acne misclassifications found in tested images.")

if __name__ == "__main__":
    test_dark_nevi_images()
