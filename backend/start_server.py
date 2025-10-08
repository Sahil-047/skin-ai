"""
Startup script for the Skin Disease Classification API
Handles model loading and server startup
"""

import uvicorn
import sys
from pathlib import Path
import os

# Add parent directory to path for training imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(parent_dir / "training"))

def check_requirements():
    """Check if all requirements are met."""
    print("🔍 Checking requirements...")
    
    # Check if model exists
    model_path = parent_dir / "training" / "best_model.pth"
    if not model_path.exists():
        print("⚠️  Warning: No trained model found at training/best_model.pth")
        print("   The API will use pretrained weights as fallback.")
    else:
        print(f"✅ Model found: {model_path}")
    
    # Check if data exists
    data_dir = parent_dir / "data"
    if not data_dir.exists():
        print("❌ Error: data directory not found!")
        print("   Please run the image migration first.")
        return False
    
    metadata_path = data_dir / "metadata_final.csv"
    if not metadata_path.exists():
        print("❌ Error: metadata_final.csv not found!")
        print("   Please run: python metadata.py")
        return False
    
    images_dir = data_dir / "images"
    if not images_dir.exists():
        print("❌ Error: data/images directory not found!")
        print("   Please run the image migration first.")
        return False
    
    image_count = len(list(images_dir.glob("*.jpg")))
    if image_count == 0:
        print("⚠️  Warning: No images found in data/images/")
        print("   The API will still work with uploaded images.")
    else:
        print(f"✅ Found {image_count} images in data/images/")
    
    print("✅ Requirements check complete!")
    return True

def start_server():
    """Start the FastAPI server."""
    
    print("\n" + "="*70)
    print("🚀 STARTING SKIN DISEASE CLASSIFICATION API")
    print("="*70)
    
    if not check_requirements():
        print("\n❌ Requirements not met. Please fix the issues above.")
        return
    
    print("\n🌐 Starting server...")
    print("   API will be available at: http://localhost:8000")
    print("   Interactive docs at: http://localhost:8000/docs")
    print("   Web interface at: http://localhost:8000/web_interface.html")
    print("\n📱 For mobile app integration:")
    print("   Base URL: http://YOUR_IP:8000")
    print("   Predict endpoint: POST /predict")
    print("   Classes endpoint: GET /classes")
    print("\n" + "="*70)
    
    try:
        uvicorn.run(
            "main:app",
            host="0.0.0.0",  # Allow external connections
            port=8000,
            reload=True,
            log_level="info",
            access_log=True
        )
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped by user.")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")

if __name__ == "__main__":
    start_server()
