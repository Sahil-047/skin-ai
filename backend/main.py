"""
FastAPI Backend for Skin Disease Classification
Provides /predict endpoint with Grad-CAM visualization
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import io
import base64
import numpy as np
import cv2
from typing import Dict, Any
import sys
from pathlib import Path

# Add training directory to path for imports
sys.path.append(str(Path(__file__).parent.parent / "training"))

from dataset import SkinDataset
from transformer import get_val_transforms

app = FastAPI(
    title="Skin Disease Classification API",
    description="AI-powered skin disease classification with Grad-CAM visualization",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (web interface)
@app.get("/web_interface.html")
async def serve_web_interface():
    """Serve the web interface HTML file."""
    return FileResponse("web_interface.html")

# Global variables
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = {
    0: 'akiec', 1: 'bcc', 2: 'bkl', 
    3: 'df', 4: 'mel', 5: 'nv', 6: 'vasc'
}
class_descriptions = {
    0: 'Actinic keratoses and intraepithelial carcinoma',
    1: 'Basal cell carcinoma',
    2: 'Benign keratosis-like lesions',
    3: 'Dermatofibroma',
    4: 'Melanoma',
    5: 'Melanocytic nevi',
    6: 'Vascular lesions'
}


class GradCAM:
    """Grad-CAM implementation for model interpretability."""
    
    def __init__(self, model, target_layer_name="layer4"):
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None
        self.hooks = []
        
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks."""
        def forward_hook(module, input, output):
            self.activations = output
            
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
            
        # Find the target layer
        for name, module in self.model.named_modules():
            if name == self.target_layer_name:
                self.hooks.append(module.register_forward_hook(forward_hook))
                self.hooks.append(module.register_backward_hook(backward_hook))
                break
    
    def generate_cam(self, input_tensor, class_idx=None):
        """Generate Grad-CAM heatmap."""
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1)
        
        # Backward pass
        self.model.zero_grad()
        output[0, class_idx].backward(retain_graph=True)
        
        # Generate CAM
        gradients = self.gradients[0].cpu().data.numpy()
        activations = self.activations[0].cpu().data.numpy()
        
        # Global average pooling of gradients
        weights = np.mean(gradients, axis=(1, 2))
        
        # Weighted combination of activation maps
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
        
        # Apply ReLU
        cam = np.maximum(cam, 0)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / cam.max() if cam.max() > 0 else cam
        
        return cam
    
    def remove_hooks(self):
        """Remove registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


def load_model():
    """Load the trained model."""
    global model
    
    try:
        # Import model architecture
        from torchvision import models
        
        # Create model
        model = models.resnet18(pretrained=False)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 7)
        
        # Load weights
        model_path = Path(__file__).parent.parent / "training" / "best_model.pth"
        if model_path.exists():
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"✅ Model loaded from {model_path}")
        else:
            # Load pretrained weights as fallback
            model = models.resnet18(pretrained=True)
            model.fc = nn.Linear(num_features, 7)
            print("⚠️  Using pretrained weights (no custom training found)")
        
        model = model.to(device)
        model.eval()
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False


def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """Preprocess uploaded image for inference."""
    try:
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Apply transforms
        transform = get_val_transforms(img_size=224)
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        
        return image_tensor.to(device)
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image preprocessing failed: {str(e)}")


def create_gradcam_overlay(image_bytes: bytes, cam: np.ndarray) -> str:
    """Create Grad-CAM overlay and return as base64."""
    try:
        # Load original image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = np.array(image)
        
        # Resize CAM to match image
        cam_resized = cv2.resize(cam, (image.shape[1], image.shape[0]))
        
        # Apply colormap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        
        # Blend with original image
        overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
        
        # Convert to PIL and then to base64
        overlay_pil = Image.fromarray(overlay)
        buffer = io.BytesIO()
        overlay_pil.save(buffer, format="JPEG")
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return img_str
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Grad-CAM overlay creation failed: {str(e)}")


@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup."""
    print("🚀 Starting Skin Disease Classification API...")
    if not load_model():
        raise RuntimeError("Failed to load model")


@app.get("/")
async def root():
    """Root endpoint - serve web interface or health check."""
    return {
        "message": "Skin Disease Classification API",
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device),
        "endpoints": {
            "health": "/",
            "classes": "/classes", 
            "predict": "/predict",
            "docs": "/docs",
            "web_interface": "/web_interface.html"
        }
    }

@app.get("/web")
async def web_interface():
    """Redirect to web interface."""
    return FileResponse("web_interface.html")


@app.get("/classes")
async def get_classes():
    """Get available disease classes."""
    return {
        "classes": [
            {
                "id": idx,
                "name": name,
                "description": class_descriptions[idx]
            }
            for idx, name in class_names.items()
        ]
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    include_gradcam: bool = True
) -> Dict[str, Any]:
    """
    Predict skin disease from uploaded image.
    
    Args:
        file: Image file (JPEG, PNG)
        include_gradcam: Whether to include Grad-CAM visualization
    
    Returns:
        JSON response with prediction results
    """
    
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image
        image_bytes = await file.read()
        
        # Preprocess
        input_tensor = preprocess_image(image_bytes)
        
        # Prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = outputs.argmax(dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Get top 3 predictions
        top3_probs, top3_indices = torch.topk(probabilities[0], 3)
        top3_predictions = [
            {
                "class_id": int(idx.item()),
                "class_name": class_names[int(idx.item())],
                "description": class_descriptions[int(idx.item())],
                "probability": float(prob.item())
            }
            for idx, prob in zip(top3_indices, top3_probs)
        ]
        
        # Generate Grad-CAM if requested
        gradcam_base64 = None
        if include_gradcam:
            try:
                gradcam = GradCAM(model, target_layer_name="layer4")
                cam = gradcam.generate_cam(input_tensor, predicted_class)
                gradcam_base64 = create_gradcam_overlay(image_bytes, cam)
                gradcam.remove_hooks()
            except Exception as e:
                print(f"⚠️  Grad-CAM failed: {e}")
                # Continue without Grad-CAM
        
        # Prepare response
        response = {
            "success": True,
            "prediction": {
                "class_id": predicted_class,
                "class_name": class_names[predicted_class],
                "description": class_descriptions[predicted_class],
                "confidence": confidence,
                "top3_predictions": top3_predictions
            },
            "metadata": {
                "model": "ResNet18",
                "device": str(device),
                "image_size": "224x224",
                "gradcam_included": gradcam_base64 is not None
            }
        }
        
        if gradcam_base64:
            response["gradcam_overlay"] = gradcam_base64
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch")
async def predict_batch(
    files: list[UploadFile] = File(...),
    include_gradcam: bool = False
) -> Dict[str, Any]:
    """
    Predict skin disease for multiple images.
    """
    
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    results = []
    
    for i, file in enumerate(files):
        try:
            # Process each image
            image_bytes = await file.read()
            input_tensor = preprocess_image(image_bytes)
            
            # Prediction
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = outputs.argmax(dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            result = {
                "filename": file.filename,
                "success": True,
                "prediction": {
                    "class_id": predicted_class,
                    "class_name": class_names[predicted_class],
                    "confidence": confidence
                }
            }
            
            results.append(result)
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })
    
    return {
        "success": True,
        "results": results,
        "total_processed": len(files),
        "successful": sum(1 for r in results if r["success"])
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
