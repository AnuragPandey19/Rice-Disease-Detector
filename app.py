"""
Rice Leaf Disease Detection - Flask Backend
Production-Ready Web Application
Version: 1.0
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import io
import base64
import os
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global variables
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# Configuration
IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 0.95

# Class names
CLASS_NAMES_STAGE1 = [
    'bacterial_leaf_blight', 'brown_spot', 'healthy', 
    'leaf_blast', 'leaf_scald', 'narrow_brown_spot', 'not_rice_leaf'
]

CLASS_NAMES_STAGE2 = [
    'bacterial_leaf_blight', 'brown_spot', 'leaf_blast', 
    'leaf_scald', 'narrow_brown_spot'
]

BACTERIAL_DISEASES = [
    'bacterial_leaf_blight', 'brown_spot', 'leaf_blast', 
    'leaf_scald', 'narrow_brown_spot'
]

# Disease information
DISEASE_INFO = {
    'healthy': {
        'name': 'Healthy Rice Leaf',
        'severity': 'None',
        'description': 'The leaf appears to be healthy with no signs of disease.',
        'recommendation': 'Continue regular monitoring and maintain good agricultural practices.',
        'icon': '✅'
    },
    'not_rice_leaf': {
        'name': 'Not a Rice Leaf',
        'severity': 'N/A',
        'description': 'The uploaded image does not appear to be a rice leaf.',
        'recommendation': 'Please upload a clear image of a rice leaf for accurate diagnosis.',
        'icon': '❌'
    },
    'bacterial_leaf_blight': {
        'name': 'Bacterial Leaf Blight',
        'severity': 'High',
        'description': 'A serious bacterial disease causing water-soaked lesions that turn yellow or white.',
        'recommendation': 'Use resistant varieties, apply copper-based bactericides, ensure proper field drainage.',
        'icon': '🦠'
    },
    'brown_spot': {
        'name': 'Brown Spot',
        'severity': 'Medium',
        'description': 'Fungal disease causing brown spots with gray centers on leaves.',
        'recommendation': 'Apply fungicides, use disease-free seeds, maintain proper nutrition.',
        'icon': '🟤'
    },
    'leaf_blast': {
        'name': 'Leaf Blast',
        'severity': 'High',
        'description': 'Fungal disease causing diamond-shaped lesions with gray centers and brown margins.',
        'recommendation': 'Use resistant varieties, apply systemic fungicides, avoid excessive nitrogen.',
        'icon': '💥'
    },
    'leaf_scald': {
        'name': 'Leaf Scald',
        'severity': 'Medium',
        'description': 'Bacterial disease causing lesions with wavy edges and yellowing.',
        'recommendation': 'Use resistant varieties, remove infected plants, apply copper bactericides.',
        'icon': '🔥'
    },
    'narrow_brown_spot': {
        'name': 'Narrow Brown Spot',
        'severity': 'Low',
        'description': 'Fungal disease causing narrow brown lesions on leaves.',
        'recommendation': 'Improve soil fertility, apply fungicides if severe, use resistant varieties.',
        'icon': '📏'
    }
}

# ============================================
# MODEL ARCHITECTURES
# ============================================

def create_efficientnet_b3(num_classes):
    """EfficientNet-B3 for Stage 1"""
    model = models.efficientnet_b3(pretrained=False)
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    return model

def create_densenet121(num_classes):
    """DenseNet-121 for Stage 1"""
    model = models.densenet121(pretrained=False)
    num_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    return model

def create_mobilenetv3_large(num_classes):
    """MobileNetV3-Large for Stage 1"""
    model = models.mobilenet_v3_large(pretrained=False)
    num_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.Hardswish(),
        nn.Dropout(0.4),
        nn.Linear(512, 256),
        nn.Hardswish(),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes)
    )
    return model

def create_vit_base(num_classes):
    """Vision Transformer for Stage 2"""
    try:
        model = models.vit_b_16(pretrained=False)
        num_features = model.heads.head.in_features
        model.heads.head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    except:
        model = models.efficientnet_b4(pretrained=False)
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    return model

def create_convnext_tiny(num_classes):
    """ConvNeXt-Tiny for Stage 2"""
    try:
        model = models.convnext_tiny(pretrained=False)
        num_features = model.classifier[2].in_features
        model.classifier[2] = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    except:
        model = models.resnext50_32x4d(pretrained=False)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    return model

# ============================================
# TWO-STAGE ENSEMBLE PREDICTOR
# ============================================

class TwoStageEnsemblePredictor:
    """Production-optimized two-stage ensemble predictor"""
    
    def __init__(self, stage1_models, stage2_models, device, confidence_threshold=0.95):
        self.stage1_models = stage1_models
        self.stage2_models = stage2_models
        self.device = device
        self.confidence_threshold = confidence_threshold
        
        # Set to eval mode
        for model in self.stage1_models.values():
            model.eval()
        for model in self.stage2_models.values():
            model.eval()
    
    def predict_stage1(self, image_tensor):
        """Stage 1: Predict all 7 categories"""
        predictions = []
        
        with torch.no_grad():
            for model in self.stage1_models.values():
                output = model(image_tensor)
                prob = torch.softmax(output, dim=1)
                predictions.append(prob.cpu().numpy())
        
        avg_prediction = np.mean(predictions, axis=0)
        predicted_class = np.argmax(avg_prediction, axis=1)[0]
        confidence = avg_prediction[0][predicted_class]
        
        return predicted_class, confidence, avg_prediction[0]
    
    def predict_stage2(self, image_tensor):
        """Stage 2: Predict specific disease"""
        predictions = []
        
        with torch.no_grad():
            for model in self.stage2_models.values():
                output = model(image_tensor)
                prob = torch.softmax(output, dim=1)
                predictions.append(prob.cpu().numpy())
        
        avg_prediction = np.mean(predictions, axis=0)
        predicted_class = np.argmax(avg_prediction, axis=1)[0]
        confidence = avg_prediction[0][predicted_class]
        
        return predicted_class, confidence, avg_prediction[0]
    
    def predict(self, image_tensor):
        """Complete two-stage prediction"""
        # Stage 1
        stage1_class, stage1_conf, stage1_probs = self.predict_stage1(image_tensor)
        stage1_label = CLASS_NAMES_STAGE1[stage1_class]
        
        result = {
            'stage1': {
                'class': stage1_label,
                'confidence': float(stage1_conf),
                'probabilities': {
                    CLASS_NAMES_STAGE1[i]: float(stage1_probs[i]) 
                    for i in range(len(CLASS_NAMES_STAGE1))
                }
            },
            'stage2_executed': False
        }
        
        # Check if bacterial disease
        is_bacterial = stage1_label in BACTERIAL_DISEASES
        
        if not is_bacterial:
            result['final_diagnosis'] = stage1_label
            result['final_confidence'] = float(stage1_conf)
            return result
        
        # Run Stage 2 if confidence < threshold
        if stage1_conf < self.confidence_threshold:
            stage2_class, stage2_conf, stage2_probs = self.predict_stage2(image_tensor)
            stage2_label = CLASS_NAMES_STAGE2[stage2_class]
            
            result['stage2'] = {
                'disease_type': stage2_label,
                'confidence': float(stage2_conf),
                'probabilities': {
                    CLASS_NAMES_STAGE2[i]: float(stage2_probs[i]) 
                    for i in range(len(CLASS_NAMES_STAGE2))
                }
            }
            result['stage2_executed'] = True
            result['final_diagnosis'] = stage2_label
            result['final_confidence'] = float(stage1_conf * stage2_conf)
        else:
            result['final_diagnosis'] = stage1_label
            result['final_confidence'] = float(stage1_conf)
        
        return result

# ============================================
# LOAD MODELS
# ============================================

def load_models():
    """Load all trained models"""
    logger.info("Loading models...")
    
    # Stage 1 models
    stage1_models = {
        'efficientnet_b3': create_efficientnet_b3(7).to(device),
        'densenet121': create_densenet121(7).to(device),
        'mobilenetv3': create_mobilenetv3_large(7).to(device)
    }
    
    # Stage 2 models (only 2 models - optimized)
    stage2_models = {
        'vit_base': create_vit_base(5).to(device),
        'convnext_tiny': create_convnext_tiny(5).to(device)
    }
    
    # Load weights
    model_dir = 'saved_models'
    
    # Load Stage 1
    for name, model in stage1_models.items():
        model_path = os.path.join(model_dir, 'stage1_models', f'{name}_*.pth')
        matching_files = [f for f in os.listdir(os.path.join(model_dir, 'stage1_models')) if f.startswith(name)]
        if matching_files:
            full_path = os.path.join(model_dir, 'stage1_models', matching_files[0])
            checkpoint = torch.load(full_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded {name}: {checkpoint['best_val_acc']:.2f}%")
    
    # Load Stage 2
    for name, model in stage2_models.items():
        matching_files = [f for f in os.listdir(os.path.join(model_dir, 'stage2_models')) if f.startswith(name)]
        if matching_files:
            full_path = os.path.join(model_dir, 'stage2_models', matching_files[0])
            checkpoint = torch.load(full_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded {name}: {checkpoint['best_val_acc']:.2f}%")
    
    logger.info("All models loaded successfully!")
    return stage1_models, stage2_models

# Initialize predictor
try:
    stage1_models, stage2_models = load_models()
    predictor = TwoStageEnsemblePredictor(
        stage1_models, stage2_models, device, CONFIDENCE_THRESHOLD
    )
    logger.info("✅ Predictor initialized successfully!")
except Exception as e:
    logger.error(f"❌ Error loading models: {e}")
    predictor = None

# ============================================
# IMAGE PREPROCESSING
# ============================================

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_image(image_bytes):
    """Preprocess image for model input"""
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    return image_tensor, image

# ============================================
# ROUTES
# ============================================

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read and preprocess image
        image_bytes = file.read()
        image_tensor, original_image = preprocess_image(image_bytes)
        
        # Run prediction
        if predictor is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        result = predictor.predict(image_tensor)
        
        # Get disease info
        diagnosis = result['final_diagnosis']
        info = DISEASE_INFO.get(diagnosis, {
            'name': diagnosis.replace('_', ' ').title(),
            'severity': 'Unknown',
            'description': 'No information available',
            'recommendation': 'Consult an expert',
            'icon': '❓'
        })
        
        # Prepare response
        response = {
            'success': True,
            'diagnosis': info['name'],
            'confidence': f"{result['final_confidence']*100:.2f}%",
            'severity': info['severity'],
            'description': info['description'],
            'recommendation': info['recommendation'],
            'icon': info['icon'],
            'stage2_used': result['stage2_executed'],
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'details': {
                'stage1_prediction': result['stage1']['class'].replace('_', ' ').title(),
                'stage1_confidence': f"{result['stage1']['confidence']*100:.2f}%",
                'stage2_prediction': result.get('stage2', {}).get('disease_type', 'N/A'),
                'models_used': 5 if result['stage2_executed'] else 3
            }
        }
        
        logger.info(f"Prediction: {info['name']} ({result['final_confidence']*100:.2f}%)")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor is not None,
        'device': str(device),
        'timestamp': datetime.now().isoformat()
    })

# ============================================
# RUN APP
# ============================================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7860))  # Hugging Face default port
    app.run(host='0.0.0.0', port=port, debug=False)