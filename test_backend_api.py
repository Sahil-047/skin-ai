"""
Test backend API directly with an image
"""

import requests
import json
from pathlib import Path

def test_backend_api():
    """Test the backend API directly."""
    
    # Find a melanocytic nevus image
    images_dir = Path("data/images")
    nv_images = list(images_dir.glob("ISIC_*.jpg"))[:5]  # Get first 5 images
    
    print("Testing backend API with melanocytic nevus images:")
    print("="*60)
    
    for img_path in nv_images:
        print(f"\nTesting: {img_path.name}")
        
        try:
            # Send image to backend
            with open(img_path, 'rb') as f:
                files = {'file': (img_path.name, f, 'image/jpeg')}
                data = {'include_gradcam': 'false', 'min_confidence': '0.3'}
                
                response = requests.post(
                    'http://localhost:8000/predict',
                    files=files,
                    data=data,
                    timeout=30
                )
            
            if response.status_code == 200:
                result = response.json()
                
                if result.get('success', False):
                    prediction = result.get('prediction', {})
                    print(f"✅ Success: {prediction.get('class_name', 'Unknown')}")
                    print(f"   Confidence: {prediction.get('confidence_percentage', 'Unknown')}")
                    print(f"   Description: {prediction.get('description', 'Unknown')}")
                    
                    # Check if it's misclassified as acne
                    if prediction.get('class_name') == 'acne':
                        print("🚨 PROBLEM: Melanocytic nevus predicted as acne!")
                else:
                    print(f"❌ Failed: {result.get('error', 'Unknown error')}")
                    print(f"   Detected as: {result.get('detected_as', 'Unknown')}")
            else:
                print(f"❌ HTTP Error: {response.status_code}")
                print(f"   Response: {response.text}")
                
        except requests.exceptions.ConnectionError:
            print("❌ Backend not running. Start it with: python start_backend.py")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n" + "="*60)
    print("If you see acne predictions above, there's a backend issue.")
    print("If not, the problem might be in your React app or image source.")

if __name__ == "__main__":
    test_backend_api()
