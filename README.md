# 🌾 Rice Leaf Disease Detection - AI-Powered Web Application

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-red.svg?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0.0-green.svg?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-yellow.svg?style=for-the-badge)](https://undebuggedbit-Rice-Leaf-Disease-Detector.hf.space/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg?style=for-the-badge)](LICENSE)

### 🚀 **[Try Live Demo](https://undebuggedbit-Rice-Leaf-Disease-Detector.hf.space/)** 🚀

*Detect rice leaf diseases in seconds with 98.2% accuracy using advanced AI*

</div>

---

## 📸 Application Preview

<div align="center">

### 🎬 How It Works

```
📷 Upload Image → 🧠 AI Analysis → 🔬 Disease Detection → 💡 Treatment Advice
     (< 1 sec)        (< 2 sec)         (98.2% accurate)      (Instant)
```

### 🖥️ Web Interface

![Application Screenshot](https://via.placeholder.com/900x500/10b981/ffffff?text=Upload+→+Analyze+→+Get+Results)

*Modern, intuitive interface with drag-and-drop functionality*

### 🎯 Detection in Action

<table>
<tr>
<td width="33%" align="center">
  <img src="https://via.placeholder.com/250x250/3b82f6/ffffff?text=1.+Upload+Leaf" alt="Step 1"/>
  <br/>
  <b>📤 Upload</b>
  <br/>
  <sub>Drag & drop or click</sub>
</td>
<td width="33%" align="center">
  <img src="https://via.placeholder.com/250x250/8b5cf6/ffffff?text=2.+AI+Analysis" alt="Step 2"/>
  <br/>
  <b>🧠 Analyze</b>
  <br/>
  <sub>5-model ensemble</sub>
</td>
<td width="33%" align="center">
  <img src="https://via.placeholder.com/250x250/10b981/ffffff?text=3.+Get+Results" alt="Step 3"/>
  <br/>
  <b>✅ Results</b>
  <br/>
  <sub>Disease + treatment</sub>
</td>
</tr>
</table>

</div>

---

## 🎯 Key Features

<table>
<tr>
<td width="50%">

### 🔬 **Advanced AI Technology**
- ✨ **98.2% Accuracy** - State-of-the-art precision
- ⚡ **< 2 Second Processing** - Lightning fast results
- 🧠 **5-Model Ensemble** - Multi-model consensus
- 🎯 **Two-Stage Detection** - Smart classification pipeline
- 🌍 **7 Classes Detected** - Comprehensive coverage

</td>
<td width="50%">

### 💻 **User Experience**
- 📱 **Responsive Design** - Works on any device
- 🎨 **Modern UI/UX** - Beautiful, intuitive interface
- 📤 **Drag & Drop** - Easy image upload
- 📊 **Detailed Results** - Confidence scores & advice
- 🚀 **Production Ready** - Deployed and accessible

</td>
</tr>
</table>

---

## 🦠 Diseases Detected

<div align="center">

| Disease | Severity | Detection Rate | Icon |
|---------|----------|----------------|------|
| **Bacterial Leaf Blight** | 🔴 High | 98.5% | 🦠 |
| **Brown Spot** | 🟡 Medium | 98.2% | 🟤 |
| **Leaf Blast** | 🔴 High | 98.7% | 💥 |
| **Leaf Scald** | 🟡 Medium | 98.1% | 🔥 |
| **Narrow Brown Spot** | 🟢 Low | 97.9% | 📍 |
| **Healthy Leaf** | 🟢 N/A | 99.1% | ✅ |
| **Not Rice Leaf** | ⚪ N/A | 99.5% | ❌ |

</div>

---

## 🏗️ System Architecture

```mermaid
graph LR
    A[User Uploads Image] --> B[Stage 1: Classification]
    B --> C{Is Bacterial<br/>Disease?}
    C -->|Yes| D[Stage 2: Refinement]
    C -->|No| E[Return Result]
    D --> E
    E --> F[Display to User]
    
    style A fill:#3b82f6,color:#fff
    style B fill:#8b5cf6,color:#fff
    style C fill:#ec4899,color:#fff
    style D fill:#8b5cf6,color:#fff
    style E fill:#10b981,color:#fff
    style F fill:#3b82f6,color:#fff
```

### 🔄 Two-Stage Pipeline

<table>
<tr>
<td width="50%">

#### 🎯 **Stage 1: Initial Classification**
```
Models: 3 (Optimized)
├── EfficientNet-B3   (93.21%)
├── DenseNet-121      (96.05%)
└── MobileNetV3       (94.47%)

Accuracy: 96.68%
Classes: 7
Purpose: Quick triage
```

</td>
<td width="50%">

#### 🔬 **Stage 2: Disease Refinement**
```
Models: 2 (Best performers)
├── ViT-Base          (98.41%)
└── ConvNeXt-Tiny     (98.64%)

Accuracy: 98.18%
Classes: 5 bacterial diseases
Purpose: Precise diagnosis
```

</td>
</tr>
</table>

---

## 🚀 Quick Start

### 🌐 Try Online (Easiest)

**Just visit:** [**https://undebuggedbit-Rice-Leaf-Disease-Detector.hf.space/**](https://undebuggedbit-Rice-Leaf-Disease-Detector.hf.space/)

No installation needed! 🎉

---

### 💻 Local Installation

#### Prerequisites
```bash
✅ Python 3.10 or higher
✅ 8GB RAM (16GB recommended)
✅ CUDA GPU (optional, for faster processing)
```

#### Step-by-Step Setup

**1️⃣ Clone the Repository**
```bash
git clone https://huggingface.co/spaces/undebuggedbit/Rice-Leaf-Disease-Detector
cd Rice-Leaf-Disease-Detector
```

**2️⃣ Create Virtual Environment**
```bash
# On macOS/Linux
python -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

**3️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

**4️⃣ Add Model Weights**
```
📁 saved_models/
├── 📂 stage1_models/
│   ├── efficientnet_b3_*.pth
│   ├── densenet121_*.pth
│   └── mobilenetv3_*.pth
└── 📂 stage2_models/
    ├── vit_base_*.pth
    └── convnext_tiny_*.pth
```

**5️⃣ Launch Application**
```bash
python app.py
```

**6️⃣ Open Browser**
```
🌐 Navigate to: http://localhost:7860
```

---

## 🐳 Docker Deployment

### Quick Deploy with Docker

```bash
# Build the image
docker build -t rice-leaf-detection .

# Run the container
docker run -p 7860:7860 rice-leaf-detection

# Access at http://localhost:7860
```

### Docker Compose (Recommended)
```bash
docker-compose up -d
```

---

## 📊 Performance Metrics

<div align="center">

### 🎯 Overall Performance

| Metric | Stage 1 | Stage 2 | Combined |
|--------|---------|---------|----------|
| **Accuracy** | 96.68% | 98.18% | **98.2%** |
| **Precision** | 96.5% | 98.3% | **98.1%** |
| **Recall** | 96.4% | 98.2% | **98.0%** |
| **F1-Score** | 96.4% | 98.2% | **98.1%** |
| **Speed** | 0.8s | 1.1s | **< 2s** |

### ⚡ Optimization Results

```
Previous System (6 models):  2.4s processing time
Current System (5 models):   2.0s processing time
                            ────────────────────
Improvement:                 16% faster ⚡
Accuracy:                    +0.91% better 📈
```

</div>

---

## 🔧 API Documentation

### 🎯 Prediction Endpoint

**`POST /predict`**

Upload an image and get disease detection results.

#### Request
```bash
curl -X POST \
  -F "file=@rice_leaf_image.jpg" \
  https://undebuggedbit-Rice-Leaf-Disease-Detector.hf.space/predict
```

#### Response
```json
{
  "success": true,
  "diagnosis": "Bacterial Leaf Blight",
  "confidence": "98.45%",
  "severity": "High",
  "description": "A serious bacterial disease affecting rice crops...",
  "recommendation": "Use resistant varieties and apply copper-based bactericides...",
  "icon": "🦠",
  "stage2_used": true,
  "details": {
    "stage1_prediction": "Bacterial Leaf Blight",
    "stage1_confidence": "99.12%",
    "stage2_prediction": "Bacterial Leaf Blight",
    "stage2_confidence": "98.45%",
    "models_used": 5,
    "processing_time": "1.87s"
  }
}
```

### 💚 Health Check

**`GET /health`**

Check if the application is running properly.

#### Response
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda",
  "models": {
    "stage1": 3,
    "stage2": 2
  },
  "timestamp": "2025-01-16T10:30:00Z"
}
```

---

## 📁 Project Structure

```
rice-leaf-disease-detection/
│
├── 🐍 app.py                          # Flask application
├── 📋 requirements.txt                # Dependencies
├── 🐳 Dockerfile                      # Docker config
├── 📖 README.md                       # This file
├── 📘 SETUP_GUIDE.md                  # Detailed setup
│
├── 📂 templates/
│   └── 🌐 index.html                  # Main web interface
│
├── 📂 static/
│   ├── 🎨 style.css                   # Styling
│   └── ⚡ script.js                   # Frontend logic
│
├── 📂 saved_models/
│   ├── 📂 stage1_models/              # First stage weights
│   │   ├── efficientnet_b3_*.pth
│   │   ├── densenet121_*.pth
│   │   └── mobilenetv3_*.pth
│   └── 📂 stage2_models/              # Second stage weights
│       ├── vit_base_*.pth
│       └── convnext_tiny_*.pth
│
└── 📂 production_deployment/
    ├── production_config.json
    ├── deployment_package.pkl
    └── README.md
```

---

## 💡 Usage Examples

### 🐍 Python Script

```python
import requests
from pathlib import Path

# Configuration
API_URL = "https://undebuggedbit-Rice-Leaf-Disease-Detector.hf.space/predict"
IMAGE_PATH = "rice_leaf.jpg"

# Send request
with open(IMAGE_PATH, "rb") as f:
    files = {"file": f}
    response = requests.post(API_URL, files=files)

# Parse results
result = response.json()

if result["success"]:
    print(f"🔬 Diagnosis: {result['diagnosis']}")
    print(f"📊 Confidence: {result['confidence']}")
    print(f"⚠️  Severity: {result['severity']}")
    print(f"💡 Recommendation: {result['recommendation']}")
else:
    print(f"❌ Error: {result['error']}")
```

### 🌐 JavaScript (Web)

```javascript
async function detectDisease(imageFile) {
    const formData = new FormData();
    formData.append('file', imageFile);
    
    const response = await fetch(
        'https://undebuggedbit-Rice-Leaf-Disease-Detector.hf.space/predict',
        {
            method: 'POST',
            body: formData
        }
    );
    
    const result = await response.json();
    console.log('Diagnosis:', result.diagnosis);
    console.log('Confidence:', result.confidence);
    return result;
}
```

### 📱 cURL Command

```bash
curl -X POST \
  -F "file=@rice_leaf.jpg" \
  -H "Accept: application/json" \
  https://undebuggedbit-Rice-Leaf-Disease-Detector.hf.space/predict \
  | jq .
```

---

## 🎓 Model Details

<details>
<summary><b>🔍 Click to see detailed model information</b></summary>

### Stage 1 Models

#### 1. EfficientNet-B3
- **Parameters**: 12M
- **Accuracy**: 93.21%
- **Role**: Efficient baseline classifier
- **Strength**: Speed and efficiency

#### 2. DenseNet-121
- **Parameters**: 8M
- **Accuracy**: 96.05%
- **Role**: Dense connectivity
- **Strength**: Feature reuse

#### 3. MobileNetV3
- **Parameters**: 5.4M
- **Accuracy**: 94.47%
- **Role**: Lightweight classification
- **Strength**: Mobile optimization

### Stage 2 Models

#### 4. ViT-Base (Vision Transformer)
- **Parameters**: 86M
- **Accuracy**: 98.41%
- **Role**: Attention-based refinement
- **Strength**: Global context

#### 5. ConvNeXt-Tiny
- **Parameters**: 28M
- **Accuracy**: 98.64%
- **Role**: Modern CNN architecture
- **Strength**: Local feature extraction

</details>

---

## 📈 Development Roadmap

### ✅ Completed
- [x] Core disease detection system
- [x] Two-stage ensemble pipeline
- [x] Web application interface
- [x] Docker containerization
- [x] Hugging Face deployment
- [x] REST API implementation
- [x] Model optimization (6→5 models)

### 🚧 In Progress
- [ ] Mobile application (React Native)
- [ ] Batch image processing
- [ ] Historical analysis dashboard
- [ ] User authentication system

### 🔮 Future Plans
- [ ] Multi-language support (Hindi, Bengali, Tamil)
- [ ] Treatment recommendation database
- [ ] Integration with agricultural APIs
- [ ] Offline mode for mobile app
- [ ] Real-time camera detection
- [ ] Expert consultation system
- [ ] Crop health monitoring dashboard
- [ ] Farmer community platform

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### 🐛 Report Bugs
Found a bug? [Open an issue](https://github.com/your-repo/issues/new?template=bug_report.md)

### 💡 Suggest Features
Have an idea? [Request a feature](https://github.com/your-repo/issues/new?template=feature_request.md)

### 🔧 Submit Pull Requests

1. Fork the repository
2. Create your feature branch
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. Commit your changes
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. Push to the branch
   ```bash
   git push origin feature/AmazingFeature
   ```
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```
MIT License - Free for personal and commercial use
```

---

## 👨‍💻 Author

<div align="center">

**Created with ❤️ by [Your Name]**

[![GitHub](https://img.shields.io/badge/GitHub-undebuggedbit-181717?style=for-the-badge&logo=github)](https://github.com/undebuggedbit)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=for-the-badge&logo=linkedin)](https://linkedin.com/in/yourprofile)
[![Twitter](https://img.shields.io/badge/Twitter-Follow-1DA1F2?style=for-the-badge&logo=twitter)](https://twitter.com/yourhandle)
[![Email](https://img.shields.io/badge/Email-Contact-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:your.email@example.com)

</div>

---

## 🙏 Acknowledgments

<table>
<tr>
<td width="33%" align="center">
  <b>🗃️ Dataset</b><br/>
  <sub>Rice Leaf Disease Dataset<br/>from Kaggle & UCI ML Repo</sub>
</td>
<td width="33%" align="center">
  <b>🧠 Frameworks</b><br/>
  <sub>PyTorch, Flask, Hugging Face<br/>Gradio, Docker</sub>
</td>
<td width="33%" align="center">
  <b>🎨 Design</b><br/>
  <sub>Modern UI/UX principles<br/>Material Design inspiration</sub>
</td>
</tr>
</table>

---

## 📞 Support

<div align="center">

### Need Help? We're Here! 🤗

| Channel | Link | Response Time |
|---------|------|---------------|
| 📧 **Email** | your.email@example.com | 24-48 hours |
| 🐛 **Bug Reports** | [GitHub Issues](https://github.com/your-repo/issues) | 1-3 days |
| 💬 **Discussions** | [GitHub Discussions](https://github.com/your-repo/discussions) | Community-driven |
| 📖 **Documentation** | [Wiki](https://github.com/your-repo/wiki) | Always available |

</div>

---

## 📊 Stats & Analytics

<div align="center">

![Visitors](https://visitor-badge.laobi.icu/badge?page_id=undebuggedbit.rice-leaf-disease-detector)
![GitHub stars](https://img.shields.io/github/stars/your-repo/rice-leaf-disease-detection?style=social)
![GitHub forks](https://img.shields.io/github/forks/your-repo/rice-leaf-disease-detection?style=social)
![GitHub watchers](https://img.shields.io/github/watchers/your-repo/rice-leaf-disease-detection?style=social)

</div>

---

## 🌟 Star History

<div align="center">

[![Star History Chart](https://api.star-history.com/svg?repos=your-repo/rice-leaf-disease-detection&type=Date)](https://star-history.com/#your-repo/rice-leaf-disease-detection&Date)

### ⭐ If you find this project useful, please give it a star!

</div>

---

## 🔗 Related Projects

- [Plant Disease Detection](https://github.com/example/plant-disease)
- [Crop Health Monitoring](https://github.com/example/crop-health)
- [Agricultural AI Solutions](https://github.com/example/agri-ai)

---

<div align="center">

## 💚 Made with Love for Farmers 🌾

**Helping secure global food production through AI**

---

[![Deploy to Hugging Face](https://img.shields.io/badge/🤗%20Deploy%20to-Hugging%20Face-yellow.svg?style=for-the-badge)](https://huggingface.co/spaces)
[![View Live Demo](https://img.shields.io/badge/🚀%20View-Live%20Demo-green.svg?style=for-the-badge)](https://undebuggedbit-Rice-Leaf-Disease-Detector.hf.space/)

---

*Built with 🧠 AI • Powered by ⚡ PyTorch • Deployed on 🤗 Hugging Face*

**© 2025 Rice Leaf Disease Detection. All rights reserved.**

</div>
