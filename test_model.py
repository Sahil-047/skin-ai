"""
Test model loading and prediction
"""

import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path

def test_model():
    """Test if the model loads correctly."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Create model
    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 9)  # 9 classes
    
    # Load weights
    model_path = Path("training/best_model.pth")
    if model_path.exists():
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print("Model loaded successfully")
            print(f"Model has {num_features} input features")
            print(f"Model has 9 output classes")
            
            # Test with dummy input
            model.eval()
            dummy_input = torch.randn(1, 3, 224, 224).to(device)
            
            with torch.no_grad():
                output = model(dummy_input)
                print("Model prediction works")
                print(f"Output shape: {output.shape}")
                print(f"Expected shape: (1, 9)")
                
                # Check if output has correct number of classes
                if output.shape[1] == 9:
                    print("Model has correct number of output classes (9)")
                else:
                    print(f"Model has wrong number of classes: {output.shape[1]}")
                    
        except Exception as e:
            print(f"Error loading model: {e}")
            
            # Check if it's a class mismatch
            if "size mismatch" in str(e):
                print("Class mismatch detected!")
                print("The saved model has different number of classes than expected")
                print("You need to retrain the model with the new dataset")
                
    else:
        print("Model file not found!")

if __name__ == "__main__":
    test_model()
