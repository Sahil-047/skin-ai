"""
Test Client for Skin Disease Classification API
Demonstrates how to use the /predict endpoint
"""

import requests
import base64
from pathlib import Path
import json


def test_api_with_local_image(image_path: str, api_url: str = "http://localhost:8000"):
    """Test the API with a local image file."""
    
    print(f"\n🧪 Testing API with image: {image_path}")
    print("="*60)
    
    # Check if image exists
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        return None
    
    try:
        # Prepare the file
        with open(image_path, "rb") as f:
            files = {"file": (Path(image_path).name, f, "image/jpeg")}
            
            # Make request
            response = requests.post(
                f"{api_url}/predict",
                files=files,
                params={"include_gradcam": True}
            )
        
        if response.status_code == 200:
            result = response.json()
            
            print("✅ Prediction successful!")
            print(f"\n📊 Results:")
            print(f"   Class: {result['prediction']['class_name']} ({result['prediction']['class_id']})")
            print(f"   Description: {result['prediction']['description']}")
            print(f"   Confidence: {result['prediction']['confidence']:.3f}")
            
            print(f"\n🏆 Top 3 Predictions:")
            for i, pred in enumerate(result['prediction']['top3_predictions'], 1):
                print(f"   {i}. {pred['class_name']}: {pred['probability']:.3f}")
            
            print(f"\n📈 Metadata:")
            print(f"   Device: {result['metadata']['device']}")
            print(f"   Grad-CAM: {'✅' if result['metadata']['gradcam_included'] else '❌'}")
            
            # Save Grad-CAM overlay if available
            if 'gradcam_overlay' in result:
                gradcam_data = base64.b64decode(result['gradcam_overlay'])
                output_path = f"gradcam_{Path(image_path).stem}.jpg"
                with open(output_path, "wb") as f:
                    f.write(gradcam_data)
                print(f"   Grad-CAM saved: {output_path}")
            
            return result
            
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"   Response: {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Make sure the API server is running")
        print("   Start with: python backend/main.py")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def test_health_check(api_url: str = "http://localhost:8000"):
    """Test the health check endpoint."""
    
    print(f"\n🏥 Testing health check...")
    
    try:
        response = requests.get(f"{api_url}/")
        if response.status_code == 200:
            result = response.json()
            print("✅ API is healthy!")
            print(f"   Status: {result['status']}")
            print(f"   Model loaded: {result['model_loaded']}")
            print(f"   Device: {result['device']}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API server")
        return False


def test_classes_endpoint(api_url: str = "http://localhost:8000"):
    """Test the classes endpoint."""
    
    print(f"\n📚 Testing classes endpoint...")
    
    try:
        response = requests.get(f"{api_url}/classes")
        if response.status_code == 200:
            result = response.json()
            print("✅ Classes endpoint working!")
            print(f"\n🏷️  Available classes:")
            for cls in result['classes']:
                print(f"   {cls['id']}: {cls['name']} - {cls['description']}")
            return True
        else:
            print(f"❌ Classes endpoint failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API server")
        return False


def main():
    """Main test function."""
    
    print("🧪 SKIN DISEASE CLASSIFICATION API TEST CLIENT")
    print("="*70)
    
    api_url = "http://localhost:8000"
    
    # Test health check
    if not test_health_check(api_url):
        print("\n❌ API server is not running. Please start it with:")
        print("   cd backend && python main.py")
        return
    
    # Test classes endpoint
    test_classes_endpoint(api_url)
    
    # Test with sample images (if available)
    sample_images = [
        "data/images/ISIC_0024306.jpg",  # Replace with actual image names
        "data/images/ISIC_0024307.jpg",
        "data/images/ISIC_0024308.jpg"
    ]
    
    print(f"\n🖼️  Testing with sample images...")
    
    for image_path in sample_images:
        if Path(image_path).exists():
            test_api_with_local_image(image_path, api_url)
            break
    else:
        print("\n💡 To test with real images:")
        print("   1. Put some .jpg images in data/images/")
        print("   2. Run: python test_client.py")
        print("   3. Or use the web interface at http://localhost:8000/docs")


if __name__ == "__main__":
    main()
