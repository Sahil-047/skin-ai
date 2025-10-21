"""
Debug model predictions on ISIC images
"""

import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path
import pandas as pd
from PIL import Image
import torchvision.transforms as T

def debug_model_predictions():
    """Debug what the model is actually predicting."""
    
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
    
    # Load metadata to get some test images
    df = pd.read_csv('data/expanded_with_acne.csv')
    
    # Test a few images from each class
    test_classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    
    for test_class in test_classes:
        print(f"\nTesting {test_class} images:")
        
        # Get a few images of this class
        class_images = df[df['dx'] == test_class].head(3)
        
        for idx, row in class_images.iterrows():
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
                    
                    print(f"  Image: {img_name}")
                    print(f"  True class: {test_class}")
                    print(f"  Predicted: {CLASS_NAMES[predicted_class]} ({confidence:.3f})")
                    print(f"  Top 3: {[(CLASS_NAMES[idx.item()], prob.item()) for idx, prob in zip(top3_indices, top3_probs)]}")
                    
                    # Check if prediction is correct
                    true_class_id = row['label_id']
                    if predicted_class == true_class_id:
                        print(f"  Result: CORRECT")
                    else:
                        print(f"  Result: WRONG (expected class {true_class_id})")
                    
                except Exception as e:
                    print(f"  Error processing {img_name}: {e}")
            else:
                print(f"  Image not found: {img_path}")

if __name__ == "__main__":
    debug_model_predictions()
