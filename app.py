"""
Flask Backend for Rice Leaf Disease Classification
File: app.py
"""

from flask import Flask, render_template, request, jsonify
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import io
import base64

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Configuration
DISEASE_TYPES = ['Bacterial leaf blight', 'Brown spot', 'Leaf smut']
INPUT_DIMENSIONS = (224, 224)
MODEL_PATH = 'best_model.pth'

# Device setup
processing_unit = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Disease information
DISEASE_INFO = {
    'Bacterial leaf blight': {
        'description': 'A serious bacterial disease that causes water-soaked lesions on leaves, leading to wilting and plant death.',
        'symptoms': ['Water-soaked lesions', 'Yellow to white leaves', 'Wilting of seedlings'],
        'treatment': ['Use resistant varieties', 'Apply copper-based bactericides', 'Improve field drainage'],
        'severity': 'High'
    },
    'Brown spot': {
        'description': 'A fungal disease causing brown spots on leaves, reducing photosynthesis and grain quality.',
        'symptoms': ['Circular brown spots', 'Yellow halos around spots', 'Reduced grain quality'],
        'treatment': ['Apply fungicides', 'Use certified seeds', 'Maintain proper nutrition'],
        'severity': 'Medium'
    },
    'Leaf smut': {
        'description': 'A fungal disease producing black powdery spores on leaves, affecting plant growth.',
        'symptoms': ['Black powdery masses', 'Linear lesions on leaves', 'Stunted growth'],
        'treatment': ['Remove infected plants', 'Use disease-free seeds', 'Apply systemic fungicides'],
        'severity': 'Medium'
    }
}

class DiseaseClassifier:
    def __init__(self, model_path, num_classes=3):
        self.model = self.load_model(model_path, num_classes)
        self.transform = self.get_transform()
        
    def load_model(self, model_path, num_classes):
        """Load the trained model"""
        network = models.resnet18(pretrained=False)
        feature_count = network.fc.in_features
        network.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feature_count, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        if os.path.exists(model_path):
            network.load_state_dict(torch.load(model_path, map_location=processing_unit))
            network = network.to(processing_unit)
            network.eval()
            print(f"Model loaded successfully from {model_path}")
        else:
            print(f"Warning: Model file {model_path} not found!")
            
        return network
    
    def get_transform(self):
        """Get image transformation pipeline"""
        return transforms.Compose([
            transforms.Resize(INPUT_DIMENSIONS),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def predict(self, image):
        """Make prediction on image"""
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Transform image
        image_tensor = self.transform(image).unsqueeze(0).to(processing_unit)
        
        # Make prediction
        with torch.no_grad():
            predictions = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(predictions, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
        
        predicted_idx = predicted_class.item()
        confidence_score = confidence.item() * 100
        
        # Get all class probabilities
        all_probs = probabilities[0].cpu().numpy()
        
        return {
            'disease': DISEASE_TYPES[predicted_idx],
            'confidence': round(confidence_score, 2),
            'probabilities': {DISEASE_TYPES[i]: round(float(all_probs[i]) * 100, 2) 
                            for i in range(len(DISEASE_TYPES))}
        }

# Initialize classifier
classifier = DiseaseClassifier(MODEL_PATH, len(DISEASE_TYPES))

@app.route('/')
def home():
    """Render home page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Read and process image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Make prediction
        result = classifier.predict(image)
        
        # Get disease information
        disease_name = result['disease']
        disease_details = DISEASE_INFO.get(disease_name, {})
        
        # Combine results
        response = {
            'success': True,
            'prediction': result,
            'disease_info': disease_details
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/about')
def about():
    """Render about page"""
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)