# Rice Leaf Disease Detection - Two-Stage Ensemble System

## Overview
This is a two-stage ensemble deep learning system for rice leaf disease detection.

### Stage 1: Initial Classification
- **Purpose**: Classify leaves as Healthy, Bacterial Disease, or Non-leaf
- **Models**: EfficientNet-B3, DenseNet-121, MobileNetV3-Large
- **Method**: Soft voting ensemble (average predictions)

### Stage 2: Disease-Specific Classification
- **Purpose**: Identify specific bacterial disease types
- **Models**: ViT/EfficientNet-B4, ConvNeXt-Tiny/ResNeXt-50, EfficientNetV2-S/EfficientNet-B5
- **Method**: Soft voting ensemble (average predictions)
- **Trigger**: Only runs if Stage 1 detects bacterial disease

## Files Structure
```
saved_models/
|-- stage1_models/
|   |-- efficientnet_b3_*.pth
|   |-- densenet121_*.pth
|   +-- mobilenetv3_*.pth
|-- stage2_models/
|   |-- vit_base_*.pth
|   |-- convnext_tiny_*.pth
|   +-- efficientnetv2_s_*.pth
|-- ensemble_deployment_package.pkl
|-- ensemble_config.json
+-- training_summary_report.txt
```

## Usage Example
```python
# Load models and create predictor
ensemble_predictor = TwoStageEnsemblePredictor(...)

# Prepare image
image = preprocess_image(your_image)

# Get prediction
result = ensemble_predictor.predict(image)

# Result structure:
{
    'stage1': {'class': 'bacterial_leaf', 'confidence': 0.95, ...},
    'stage2': {'disease_type': 'bacterial_blight', 'confidence': 0.88, ...},
    'final_diagnosis': 'Bacterial Disease: bacterial_blight',
    'final_confidence': 0.836
}
```

## Performance Metrics
- See training_summary_report.txt for detailed metrics
- Confusion matrices saved as PNG files
- Training curves saved as PNG files

## Requirements
- PyTorch >= 1.9.0
- torchvision >= 0.10.0
- Python >= 3.7

## Notes
- All models use transfer learning with pretrained ImageNet weights
- Image preprocessing: Resize(224, 224) + Normalize
- Ensemble method: Soft voting (average probabilities)
