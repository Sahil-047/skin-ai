"""
Debug melanocytic nevi predictions specifically
"""

import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path
import pandas as pd
from PIL import Image
import torchvision.transforms as T

def debug_nv_predictions():
    """Debug what's happening with melanocytic nevi predictions."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load model
    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 9)
    
    model_path = Path("training/best_model.pth")
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print("Model loaded successfully")
    else:
        print("Model not found!")
        return
    
    # Class names
    CLASS_NAMES = {
        0: 'akiec', 1: 'bcc', 2: 'bkl', 
        3: 'df', 4: 'mel', 5: 'nv', 6: 'vasc',
        7: 'non_skin', 8: 'acne'
    }
    
    # Load metadata
    df = pd.read_csv('data/expanded_with_acne.csv')
    
    # Test melanocytic nevi images specifically
    print("\nTesting melanocytic nevi (nv) images:")
    print("="*60)
    
    nv_images = df[df['dx'] == 'nv'].head(10)  # Test more images
    
    acne_predictions = 0
    correct_predictions = 0
    
    for idx, row in nv_images.iterrows():
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
                
                print(f"Image: {img_name}")
                print(f"True class: nv (melanocytic nevi)")
                print(f"Predicted: {CLASS_NAMES[predicted_class]} ({confidence:.3f})")
                print(f"Top 3: {[(CLASS_NAMES[idx.item()], prob.item()) for idx, prob in zip(top3_indices, top3_probs)]}")
                
                # Check if prediction is correct
                if predicted_class == 5:  # nv class
                    print(f"Result: CORRECT")
                    correct_predictions += 1
                elif predicted_class == 8:  # acne class
                    print(f"Result: WRONG - Predicted as ACNE!")
                    acne_predictions += 1
                else:
                    print(f"Result: WRONG - Predicted as {CLASS_NAMES[predicted_class]}")
                
                print("-" * 40)
                
            except Exception as e:
                print(f"Error processing {img_name}: {e}")
        else:
            print(f"Image not found: {img_path}")
    
    print(f"\nSummary:")
    print(f"Total nv images tested: {len(nv_images)}")
    print(f"Correctly predicted as nv: {correct_predictions}")
    print(f"Wrongly predicted as acne: {acne_predictions}")
    print(f"Accuracy: {correct_predictions/len(nv_images)*100:.1f}%")
    
    if acne_predictions > 0:
        print(f"\nWARNING: {acne_predictions} melanocytic nevi were predicted as acne!")
        print("This is a serious issue for medical diagnosis.")
        print("Possible causes:")
        print("1. Insufficient training data for acne class")
        print("2. Model overfitting to acne features")
        print("3. Class imbalance (only 34 acne vs 6705 nv images)")

if __name__ == "__main__":
    debug_nv_predictions()
