# Rice Leaf Disease Detection - Two-Stage Ensemble System

## Overview
This is a two-stage ensemble deep learning system for rice leaf disease detection.

> **Stale-doc warning (M-11, now corrected).** This file previously described a
> 3-model Stage 2 including EfficientNetV2-S, and called Stage 2 "bacterial
> disease classification". Production has used 2 models since November 2025, and
> only one of the five diseases is bacterial. Corrected below.

### Stage 1: Initial Classification (7 classes)
- **Purpose**: Classify into 5 diseases, healthy, or not-a-rice-leaf
- **Models**: EfficientNet-B3, DenseNet-121, MobileNetV3-Large
- **Method**: Weighted soft voting (members weighted by validation accuracy)

### Stage 2: Disease Refinement (5 classes)
- **Purpose**: Re-classify among the 5 disease classes
- **Models**: ViT-Base, ConvNeXt-Tiny — **2 models**. EfficientNetV2-S was
  dropped (94.09% standalone); removing it improved the ensemble by +0.91%.
  Its checkpoints remain on disk but are not loaded by `app.py`.
- **Method**: Weighted soft voting
- **Trigger**: Stage 1 predicts **any disease** AND Stage 1 confidence < 0.95.
  Four of the five diseases are fungal, not bacterial.
- **Abstains**: if Stage 2 confidence < 0.45 the API returns `uncertain` rather
  than forcing a disease label (M-13).

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
