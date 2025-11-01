"""
Skin Disease Classification API - Single File Backend
FastAPI server with predictive model for React web app integration

Run with: python start_backend.py
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models
from PIL import Image
import io
import base64
import numpy as np
import cv2
from typing import Dict, Any, Optional
import uvicorn
from pathlib import Path
import sys


# ============================================================================
# Configuration
# ============================================================================

MODEL_PATH = Path("training/best_model.pth")
NUM_CLASSES = 15 
IMG_SIZE = 224
HOST = "0.0.0.0"  # Allow external connections
PORT = 8000

# Class definitions
CLASS_NAMES = {
    0: 'akiec', 1: 'bcc', 2: 'bkl', 
    3: 'df', 4: 'mel', 5: 'nv', 6: 'vasc',
    7: 'non_skin', 8: 'acne', 9: 'athletes_foot',
    10: 'cellulitis', 11: 'cold_sores', 12: 'eczema',
    13: 'fungal_infection', 14: 'heat_rash'
}

CLASS_DESCRIPTIONS = {
    0: 'Actinic keratoses and intraepithelial carcinoma',
    1: 'Basal cell carcinoma',
    2: 'Benign keratosis-like lesions',
    3: 'Dermatofibroma',
    4: 'Melanoma',
    5: 'Melanocytic nevi',
    6: 'Vascular lesions',
    7: 'Non-skin image (e.g., object, background)',
    8: 'Acne vulgaris (common acne)',
    9: 'Athlete\'s foot (tinea pedis)',
    10: 'Cellulitis (bacterial skin infection)',
    11: 'Cold sores (herpes simplex)',
    12: 'Eczema (atopic dermatitis)',
    13: 'Fungal infection (tinea)',
    14: 'Heat rash (prickly heat)'
}


# ============================================================================
# FastAPI Application Setup
# ============================================================================

app = FastAPI(
    title="Skin Disease Classification API",
    description="AI-powered skin disease classification with Grad-CAM visualization",
    version="2.0.0"
)

# CORS Configuration for React
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "*"  # Allow all origins - restrict in production
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


# ============================================================================
# Global Variables
# ============================================================================

model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# Image Transforms
# ============================================================================

def get_val_transforms(img_size=224):
    """Get validation/inference transforms."""
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ============================================================================
# Grad-CAM Implementation
# ============================================================================

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
        if cam.max() > 0:
            cam = cam - cam.min()
            cam = cam / cam.max()
        
        return cam
    
    def remove_hooks(self):
        """Remove registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


# ============================================================================
# Model Loading
# ============================================================================

def load_model():
    """Load the trained model."""
    global model
    
    try:
        print("🔧 Loading model...")
        
        # Create ResNet50 model (matches train_improved.py)
        model = models.resnet50(weights=None)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, NUM_CLASSES)
        
        # Load trained weights if available
        if MODEL_PATH.exists():
            state_dict = torch.load(MODEL_PATH, map_location=device)
            model.load_state_dict(state_dict)
            print(f"✅ Model loaded from {MODEL_PATH}")
            print(f"   Architecture: ResNet50")
            print(f"   Parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
        else:
            # Use pretrained weights as fallback
            print("⚠️  No custom model found, using pretrained weights")
            model = models.resnet50(weights='IMAGENET1K_V2')
            model.fc = nn.Linear(num_features, NUM_CLASSES)
        
        model = model.to(device)
        model.eval()
        print(f"✅ Model ready on {device}")
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# Skin Detection
# ============================================================================

def detect_skin(image_bytes: bytes) -> tuple[bool, float]:
    """
    Detect if image contains skin using color analysis.
    Returns (is_skin, skin_percentage)
    """
    try:
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_np = np.array(image)
        
        # Convert to HSV for better skin color detection
        hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
        
        # Define multiple skin color ranges for different skin tones
        skin_ranges = [
            # Light skin tones
            ([0, 20, 70], [20, 255, 255]),
            ([0, 30, 60], [20, 150, 255]),
            # Medium skin tones  
            ([0, 40, 50], [20, 120, 255]),
            ([0, 50, 40], [20, 100, 255]),
            # Dark skin tones
            ([0, 60, 30], [20, 80, 255]),
            ([0, 70, 20], [20, 60, 255]),
        ]
        
        # Calculate total skin pixels across all ranges
        total_skin_pixels = 0
        for lower, upper in skin_ranges:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            total_skin_pixels += cv2.countNonZero(mask)
        
        # Calculate skin percentage
        total_pixels = image_np.shape[0] * image_np.shape[1]
        skin_percentage = total_skin_pixels / total_pixels
        
        # Threshold for skin detection (adjustable)
        SKIN_THRESHOLD = 0.08  # 8% of pixels must be skin-colored
        is_skin = skin_percentage > SKIN_THRESHOLD
        
        return is_skin, skin_percentage
        
    except Exception as e:
        print(f"⚠️  Skin detection failed: {e}")
        return False, 0.0


def validate_prediction(confidence: float, min_confidence: float = 0.6) -> tuple[bool, str]:
    """
    Validate prediction confidence.
    Returns (is_valid, error_message)
    """
    if confidence < min_confidence:
        return False, f"Low confidence prediction ({confidence:.1%}). Please upload a clearer image of the skin lesion."
    return True, ""


# ============================================================================
# Image Processing
# ============================================================================

def preprocess_image(image_bytes: bytes) -> tuple[torch.Tensor, Image.Image]:
    """
    Preprocess uploaded image for inference.
    Returns both the tensor and the original PIL image.
    """
    try:
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Apply transforms
        transform = get_val_transforms(img_size=IMG_SIZE)
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        
        return image_tensor.to(device), image
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image preprocessing failed: {str(e)}")


def create_gradcam_overlay(image: Image.Image, cam: np.ndarray) -> str:
    """Create Grad-CAM overlay and return as base64."""
    try:
        # Convert PIL to numpy
        image_np = np.array(image)
        
        # Resize CAM to match image
        cam_resized = cv2.resize(cam, (image_np.shape[1], image_np.shape[0]))
        
        # Apply colormap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Blend with original image
        overlay = cv2.addWeighted(image_np, 0.6, heatmap, 0.4, 0)
        
        # Convert to base64
        overlay_pil = Image.fromarray(overlay)
        buffer = io.BytesIO()
        overlay_pil.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return f"data:image/png;base64,{img_str}"
        
    except Exception as e:
        print(f"⚠️  Grad-CAM overlay creation failed: {e}")
        return None


# ============================================================================
# API Endpoints
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup."""
    print("\n" + "="*70)
    print("🚀 STARTING SKIN DISEASE CLASSIFICATION API")
    print("="*70)
    
    if not load_model():
        print("❌ Failed to load model - API may not work correctly")
    
    print(f"\n🌐 Server starting on http://{HOST}:{PORT}")
    print(f"   • Health Check: http://localhost:{PORT}/")
    print(f"   • API Docs: http://localhost:{PORT}/docs")
    print(f"   • Classes Info: http://localhost:{PORT}/classes")
    print(f"   • Predict Endpoint: http://localhost:{PORT}/predict")
    print("\n🛡️  NEW FEATURES:")
    print("   • Skin detection - rejects non-skin images")
    print("   • Confidence filtering - rejects low-confidence predictions")
    print("   • Better error messages - explains why images are rejected")
    print("\n📱 For React integration:")
    print(f"   • Base URL: http://localhost:{PORT}")
    print(f"   • CORS enabled for localhost:3000, localhost:5173")
    print("="*70 + "\n")


@app.get("/")
async def root():
    """Health check and API info."""
    return {
        "status": "healthy",
        "message": "Skin Disease Classification API",
        "version": "2.0.0",
        "model_loaded": model is not None,
        "device": str(device),
        "endpoints": {
            "health": "/",
            "classes": "/classes",
            "predict": "/predict (POST)",
            "batch_predict": "/predict/batch (POST)",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "model_path_exists": MODEL_PATH.exists()
    }


@app.get("/classes")
async def get_classes():
    """Get available disease classes."""
    return {
        "classes": [
            {
                "id": idx,
                "name": name,
                "description": CLASS_DESCRIPTIONS[idx]
            }
            for idx, name in CLASS_NAMES.items()
        ]
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    include_gradcam: bool = True,
    min_confidence: float = 0.6
) -> Dict[str, Any]:
    """
    Predict skin disease from uploaded image with skin detection and confidence filtering.
    
    Args:
        file: Image file (JPEG, PNG, etc.)
        include_gradcam: Whether to include Grad-CAM visualization (default: True)
        min_confidence: Minimum confidence threshold (default: 0.6)
    
    Returns:
        JSON response with prediction results or rejection reason
    """
    
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please check server logs."
        )
    
    try:
        # Validate file type
        if file.content_type and not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid file type: {file.content_type}. Must be an image."
            )
        
        # Read image
        image_bytes = await file.read()
        
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty file received")
        
        # SKIN DETECTION CHECK
        is_skin, skin_percentage = detect_skin(image_bytes)
        
        if not is_skin:
            return {
                "success": False,
                "error": "Image does not appear to contain skin",
                "skin_percentage": f"{skin_percentage:.1%}",
                "suggestion": "Please upload a clear image of a skin lesion",
                "detected_as": "non_skin",
                "metadata": {
                    "filename": file.filename,
                    "skin_detection": "failed",
                    "skin_percentage": skin_percentage
                }
            }
        
        # Preprocess
        input_tensor, original_image = preprocess_image(image_bytes)
        
        # Prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = outputs.argmax(dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # NON-SKIN CLASS CHECK
        if predicted_class == 7:  # non_skin class
            return {
                "success": False,
                "error": "Image does not appear to be a skin lesion",
                "confidence": confidence,
                "confidence_percentage": f"{confidence:.1%}",
                "suggestion": "Please upload a clear image of a skin lesion",
                "detected_as": "non_skin",
                "metadata": {
                    "filename": file.filename,
                    "skin_detection": "passed",
                    "skin_percentage": skin_percentage,
                    "model_prediction": "non_skin",
                    "confidence_threshold": min_confidence
                }
            }
        
        # CONFIDENCE VALIDATION
        is_valid, error_msg = validate_prediction(confidence, min_confidence)
        
        if not is_valid:
            return {
                "success": False,
                "error": error_msg,
                "confidence": confidence,
                "confidence_percentage": f"{confidence:.1%}",
                "suggestion": "Please upload a clearer image of the skin lesion",
                "detected_as": "uncertain",
                "metadata": {
                    "filename": file.filename,
                    "skin_detection": "passed",
                    "skin_percentage": skin_percentage,
                    "confidence_threshold": min_confidence
                }
            }
        
        # Get top 3 predictions
        top3_probs, top3_indices = torch.topk(probabilities[0], min(3, NUM_CLASSES))
        top3_predictions = [
            {
                "class_id": int(idx.item()),
                "class_name": CLASS_NAMES[int(idx.item())],
                "description": CLASS_DESCRIPTIONS[int(idx.item())],
                "probability": float(prob.item()),
                "percentage": f"{float(prob.item()) * 100:.2f}%"
            }
            for idx, prob in zip(top3_indices, top3_probs)
        ]
        
        # Generate Grad-CAM if requested
        gradcam_overlay = None
        if include_gradcam:
            try:
                gradcam = GradCAM(model, target_layer_name="layer4")
                cam = gradcam.generate_cam(input_tensor, predicted_class)
                gradcam_overlay = create_gradcam_overlay(original_image, cam)
                gradcam.remove_hooks()
            except Exception as e:
                print(f"⚠️  Grad-CAM generation failed: {e}")
                # Continue without Grad-CAM
        
        # Prepare response
        response = {
            "success": True,
            "prediction": {
                "class_id": predicted_class,
                "class_name": CLASS_NAMES[predicted_class],
                "description": CLASS_DESCRIPTIONS[predicted_class],
                "confidence": confidence,
                "confidence_percentage": f"{confidence * 100:.2f}%",
                "top3_predictions": top3_predictions
            },
            "metadata": {
                "model": "ResNet50",
                "device": str(device),
                "image_size": f"{IMG_SIZE}x{IMG_SIZE}",
                "gradcam_included": gradcam_overlay is not None,
                "filename": file.filename,
                "skin_detection": "passed",
                "skin_percentage": f"{skin_percentage:.1%}",
                "confidence_threshold": min_confidence,
                "validation": "passed"
            }
        }
        
        if gradcam_overlay:
            response["gradcam_overlay"] = gradcam_overlay
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500, 
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch")
async def predict_batch(
    files: list[UploadFile] = File(...),
    include_gradcam: bool = False
) -> Dict[str, Any]:
    """
    Predict skin disease for multiple images.
    Grad-CAM is disabled by default for performance.
    """
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(files) > 50:
        raise HTTPException(
            status_code=400, 
            detail="Maximum 50 images per batch request"
        )
    
    results = []
    
    for i, file in enumerate(files):
        try:
            # Process each image
            image_bytes = await file.read()
            input_tensor, _ = preprocess_image(image_bytes)
            
            # Prediction
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = outputs.argmax(dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            result = {
                "index": i,
                "filename": file.filename,
                "success": True,
                "prediction": {
                    "class_id": predicted_class,
                    "class_name": CLASS_NAMES[predicted_class],
                    "description": CLASS_DESCRIPTIONS[predicted_class],
                    "confidence": confidence,
                    "confidence_percentage": f"{confidence * 100:.2f}%"
                }
            }
            
            results.append(result)
            
        except Exception as e:
            results.append({
                "index": i,
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })
    
    return {
        "success": True,
        "results": results,
        "total_processed": len(files),
        "successful": sum(1 for r in results if r.get("success", False)),
        "failed": sum(1 for r in results if not r.get("success", True))
    }


# ============================================================================
# Server Startup
# ============================================================================

def main():
    """Start the FastAPI server."""
    print("="*70)
    print("SKIN DISEASE CLASSIFICATION API")
    print("="*70)
    print("Model: ResNet50 (15 classes)")
    print("Classes: akiec, bcc, bkl, df, mel, nv, vasc, non_skin, acne, athletes_foot, cellulitis, cold_sores, eczema, fungal_infection, heat_rash")
    print("Features: Skin detection, Confidence filtering, Grad-CAM")
    print("Training: Weighted loss for class imbalance")
    print("="*70)
    print(f"Starting server on http://{HOST}:{PORT}")
    print("="*70)
    
    try:
        uvicorn.run(
            app,
            host=HOST,
            port=PORT,
            log_level="info",
            access_log=True
        )
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped by user.")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
