# 🌾 Rice Leaf Disease Detection - AI-Powered Web Application

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-red.svg)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0.0-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A production-ready web application for detecting rice leaf diseases using advanced deep learning models with 98.2% accuracy.

![Demo](https://via.placeholder.com/800x400/10b981/ffffff?text=Rice+Leaf+Disease+Detection)

---

## 🎯 Features

- **🔬 High Accuracy**: 98.2% detection accuracy using ensemble deep learning
- **⚡ Fast Processing**: Results in under 2 seconds
- **🧠 Smart Two-Stage System**: Optimized 5-model ensemble
- **🌿 7-Class Detection**: Healthy, non-leaf, and 5 bacterial diseases
- **📱 Responsive UI**: Beautiful, modern interface works on all devices
- **🚀 Production Ready**: Docker support, easy deployment

---

## 📊 Model Performance

### Stage 1: Initial Classification (7 classes)
- **Accuracy**: 96.68%
- **Models**: EfficientNet-B3, DenseNet-121, MobileNetV3
- **Classes**: 
  - Healthy
  - Not Rice Leaf
  - 5 Bacterial Diseases

### Stage 2: Disease Refinement (5 classes)
- **Accuracy**: 98.18%
- **Models**: ViT-Base, ConvNeXt-Tiny (optimized from 3 to 2 models)
- **Diseases Detected**:
  1. Bacterial Leaf Blight
  2. Brown Spot
  3. Leaf Blast
  4. Leaf Scald
  5. Narrow Brown Spot

---

## 🏗️ Architecture

```
User Upload → Stage 1 (3 models) → Classification
                ↓
        Is it bacterial disease?
                ↓
        Yes → Stage 2 (2 models) → Refined Diagnosis
        No  → Return Result
```

### Models Used:
1. **EfficientNet-B3** (Stage 1) - 93.21%
2. **DenseNet-121** (Stage 1) - 96.05%
3. **MobileNetV3** (Stage 1) - 94.47%
4. **ViT-Base** (Stage 2) - 98.41%
5. **ConvNeXt-Tiny** (Stage 2) - 98.64%

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (optional, but recommended)
- 8GB RAM minimum

### Installation

1. **Clone the repository**
```bash
git clone https://huggingface.co/spaces/your-username/rice-leaf-disease-detection
cd rice-leaf-disease-detection
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Add your trained models**
```
Place your .pth model files in:
saved_models/
├── stage1_models/
│   ├── efficientnet_b3_*.pth
│   ├── densenet121_*.pth
│   └── mobilenetv3_*.pth
└── stage2_models/
    ├── vit_base_*.pth
    └── convnext_tiny_*.pth
```

5. **Run the application**
```bash
python app.py
```

6. **Open browser**
```
Navigate to: http://localhost:7860
```

---

## 🐳 Docker Deployment

### Build Docker Image
```bash
docker build -t rice-leaf-detection .
```

### Run Container
```bash
docker run -p 7860:7860 rice-leaf-detection
```

---

## 🌐 Deploy to Hugging Face Spaces

1. **Create a new Space** on [Hugging Face](https://huggingface.co/spaces)
2. **Select**: Docker as the Space SDK
3. **Upload files**:
   - All project files
   - Model files in `saved_models/`
4. **Space will auto-deploy**
5. **Access your app** at: `https://huggingface.co/spaces/your-username/your-space-name`

---

## 📁 Project Structure

```
rice-leaf-disease-detection/
├── app.py                      # Flask backend
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker configuration
├── .gitignore                  # Git ignore rules
├── README.md                   # This file
├── SETUP_GUIDE.md             # Detailed setup instructions
│
├── templates/
│   └── index.html             # Frontend HTML
│
├── static/
│   ├── style.css              # Styling
│   └── script.js              # JavaScript logic
│
├── saved_models/
│   ├── stage1_models/         # Stage 1 model weights
│   │   ├── efficientnet_b3_*.pth
│   │   ├── densenet121_*.pth
│   │   └── mobilenetv3_*.pth
│   └── stage2_models/         # Stage 2 model weights
│       ├── vit_base_*.pth
│       └── convnext_tiny_*.pth
│
└── production_deployment/     # Deployment configs
    ├── production_config.json
    ├── deployment_package.pkl
    └── README.md
```

---

## 🔧 API Endpoints

### `POST /predict`
Upload an image for disease detection.

**Request:**
```bash
curl -X POST -F "file=@image.jpg" http://localhost:7860/predict
```

**Response:**
```json
{
  "success": true,
  "diagnosis": "Bacterial Leaf Blight",
  "confidence": "98.45%",
  "severity": "High",
  "description": "A serious bacterial disease...",
  "recommendation": "Use resistant varieties...",
  "icon": "🦠",
  "stage2_used": true,
  "details": {
    "stage1_prediction": "Bacterial Leaf Blight",
    "stage1_confidence": "99.12%",
    "models_used": 5
  }
}
```

### `GET /health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda",
  "timestamp": "2025-01-16T10:30:00"
}
```

---

## 🎨 Frontend Features

- **Drag & Drop Upload**: Intuitive image upload
- **Real-time Preview**: See uploaded image instantly
- **Animated Results**: Beautiful result presentation
- **Responsive Design**: Works on mobile, tablet, desktop
- **Dark/Light Mode**: (Coming soon)

---

## 📈 Performance Optimization

### Model Optimization
- ✅ Reduced from 6 to 5 models (removed underperforming EfficientNetV2-S)
- ✅ Conditional Stage 2 execution (only when needed)
- ✅ 16% faster inference time
- ✅ +0.91% accuracy improvement

### Production Improvements
- Gunicorn WSGI server
- Model caching
- Image preprocessing optimization
- Async processing ready

---

## 🛠️ Development

### Run in Development Mode
```bash
export FLASK_ENV=development
export FLASK_DEBUG=1
python app.py
```

### Run Tests
```bash
pytest tests/
```

### Code Quality
```bash
# Format code
black app.py

# Lint
flake8 app.py

# Type check
mypy app.py
```

---

## 📝 Usage Examples

### Python Script
```python
import requests

url = "http://localhost:7860/predict"
files = {"file": open("rice_leaf.jpg", "rb")}

response = requests.post(url, files=files)
result = response.json()

print(f"Diagnosis: {result['diagnosis']}")
print(f"Confidence: {result['confidence']}")
```

### cURL
```bash
curl -X POST \
  -F "file=@rice_leaf.jpg" \
  http://localhost:7860/predict
```

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Authors

- **Your Name** - Initial work

---

## 🙏 Acknowledgments

- Dataset: Rice Leaf Disease Dataset
- Models: PyTorch Model Zoo
- Framework: Flask
- UI Inspiration: Modern web design trends

---

## 📞 Support

For issues and questions:
- 📧 Email: your.email@example.com
- 🐛 Issues: [GitHub Issues](https://github.com/your-repo/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/your-repo/discussions)

---

## 🔮 Future Enhancements

- [ ] Mobile app (React Native)
- [ ] Batch processing
- [ ] Historical analysis dashboard
- [ ] Multi-language support
- [ ] Treatment recommendations database
- [ ] Integration with agricultural APIs

---

## ⭐ Star History

If you find this project useful, please consider giving it a star!

---

**Made with ❤️ and 🧠 AI**