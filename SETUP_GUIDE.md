# 🚀 Complete Setup Guide - Rice Leaf Disease Detection

This guide will walk you through deploying your Rice Leaf Disease Detection application to Hugging Face Spaces.

---

## 📋 Prerequisites Checklist

- [ ] Trained model files (.pth files)
- [ ] Hugging Face account (free)
- [ ] Git installed (optional, can use web interface)
- [ ] All project files from this package

---

## 🎯 Step-by-Step Deployment to Hugging Face

### Step 1: Prepare Your Model Files

1. **Locate your trained models** from Jupyter notebook:
   ```
   saved_models/
   ├── stage1_models/
   │   ├── efficientnet_b3_20251115_104218.pth
   │   ├── densenet121_20251115_110746.pth
   │   └── mobilenetv3_20251115_113045.pth
   └── stage2_models/
       ├── vit_base_20251116_002651.pth
       └── convnext_tiny_20251116_004720.pth
   ```

2. **Copy these files** to your project folder maintaining the structure

---

### Step 2: Create Hugging Face Space

1. **Go to** [Hugging Face Spaces](https://huggingface.co/spaces)

2. **Click** "Create new Space"

3. **Fill in details**:
   - Space name: `rice-leaf-disease-detection`
   - License: `MIT`
   - Select SDK: **Docker** (Important!)
   - Make it: Public or Private (your choice)

4. **Click** "Create Space"

---

### Step 3: Upload Files to Hugging Face

#### Option A: Using Web Interface (Easiest)

1. **Click** "Files" tab in your Space

2. **Click** "Add file" → "Upload files"

3. **Upload these files in order**:
   
   **First batch** (Python files):
   ```
   - app.py
   - requirements.txt
   - Dockerfile
   - .gitignore
   - README.md
   - SETUP_GUIDE.md
   ```

4. **Create folders** and upload:
   
   **templates** folder:
   ```
   Click "Add file" → "Create a new file"
   Name it: templates/index.html
   Paste the HTML content
   Commit
   ```

   **static** folder:
   ```
   Create: static/style.css (paste CSS)
   Create: static/script.js (paste JavaScript)
   ```

   **saved_models** folder:
   ```
   Create: saved_models/stage1_models/.gitkeep
   Upload all Stage 1 .pth files
   
   Create: saved_models/stage2_models/.gitkeep
   Upload all Stage 2 .pth files
   ```

5. **Commit each upload** with message like "Add application files"

#### Option B: Using Git (Advanced)

```bash
# Clone your space
git clone https://huggingface.co/spaces/your-username/rice-leaf-disease-detection
cd rice-leaf-disease-detection

# Add all files
cp -r /path/to/your/files/* .

# Commit and push
git add .
git commit -m "Initial commit: Rice Leaf Disease Detection"
git push
```

---

### Step 4: Configure Space Settings

1. **Go to** "Settings" tab

2. **Set Python version**: 3.10

3. **Set Hardware**:
   - Free tier: CPU Basic (works but slower)
   - Upgrade (optional): CPU Upgrade or T4 GPU (faster)

4. **Set timeout**: 300 seconds (for model loading)

5. **Save settings**

---

### Step 5: Wait for Build

1. **Check** "Logs" tab to see build progress

2. **Wait** for Docker container to build (5-10 minutes first time)

3. **Look for**:
   ```
   Building...
   Installing dependencies...
   Loading models...
   ✅ Predictor initialized successfully!
   Running on http://0.0.0.0:7860
   ```

4. **If successful**, you'll see "Running" status

---

### Step 6: Test Your Application

1. **Click** "App" tab to view your live application

2. **Upload a test image** of a rice leaf

3. **Verify**:
   - Image uploads successfully
   - Analysis completes in < 5 seconds
   - Results display correctly
   - Confidence scores show
   - Recommendations appear

---

## 🔧 Troubleshooting

### Problem: "Model files not found"

**Solution**:
```bash
# Verify file structure in your Space:
saved_models/
├── stage1_models/
│   ├── efficientnet_b3_*.pth  ✓
│   ├── densenet121_*.pth      ✓
│   └── mobilenetv3_*.pth      ✓
└── stage2_models/
    ├── vit_base_*.pth         ✓
    └── convnext_tiny_*.pth    ✓
```

### Problem: "Out of memory"

**Solution**:
1. Upgrade to CPU Upgrade or T4 GPU tier
2. Or reduce batch processing in app.py

### Problem: "Build failed"

**Solution**:
1. Check Dockerfile syntax
2. Verify requirements.txt has correct versions
3. Check logs for specific error
4. Ensure all files are uploaded

### Problem: "Slow inference"

**Solution**:
1. Use GPU hardware (T4)
2. Enable model quantization
3. Reduce image size in preprocessing

### Problem: "Port binding error"

**Solution**:
Hugging Face uses port 7860 by default. Ensure app.py has:
```python
port = int(os.environ.get('PORT', 7860))
```

---

## 📊 File Size Optimization

### Model Files Are Large?

**Option 1**: Use Git LFS (Large File Storage)
```bash
git lfs install
git lfs track "*.pth"
git add .gitattributes
git add saved_models/
git commit -m "Add model files with LFS"
git push
```

**Option 2**: Upload to Hugging Face Models Hub
1. Create a model repository
2. Upload .pth files there
3. Modify app.py to download from hub:
```python
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="your-username/rice-leaf-models",
    filename="efficientnet_b3.pth"
)
```

---

## 🎨 Customization

### Change Port
Edit `app.py`:
```python
port = int(os.environ.get('PORT', 8080))  # Change 7860 to 8080
```

### Modify UI Colors
Edit `static/style.css`:
```css
:root {
    --primary: #10b981;  /* Change to your color */
    --secondary: #3b82f6;
}
```

### Add More Diseases
1. Update `CLASS_NAMES_STAGE1` and `CLASS_NAMES_STAGE2` in app.py
2. Add disease info to `DISEASE_INFO` dictionary
3. Retrain models with new classes

---

## 🚀 Performance Tuning

### Enable Model Quantization
```python
# In app.py, after loading models:
model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

### Enable Caching
```python
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_predict(image_hash):
    # Your prediction logic
    pass
```

### Use Gunicorn Workers
Edit Dockerfile:
```dockerfile
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "2", "--timeout", "120", "app:app"]
```

---

## 📱 Adding Features

### Add History Tracking
```python
# Store predictions in a database
from datetime import datetime

predictions_history = []

@app.route('/predict', methods=['POST'])
def predict():
    result = predictor.predict(image_tensor)
    
    # Add to history
    predictions_history.append({
        'timestamp': datetime.now(),
        'diagnosis': result['final_diagnosis'],
        'confidence': result['final_confidence']
    })
    
    return jsonify(result)

@app.route('/history')
def get_history():
    return jsonify(predictions_history)
```

### Add Download Results
```python
from flask import send_file
import json

@app.route('/download-result')
def download_result():
    # Create PDF or JSON
    result_json = json.dumps(last_result, indent=2)
    
    with open('result.json', 'w') as f:
        f.write(result_json)
    
    return send_file('result.json', as_attachment=True)
```

---

## 🔒 Security

### Add API Key Authentication
```python
from functools import wraps

API_KEY = os.environ.get('API_KEY', 'your-secret-key')

def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        key = request.headers.get('X-API-Key')
        if key != API_KEY:
            return jsonify({'error': 'Invalid API key'}), 401
        return f(*args, **kwargs)
    return decorated

@app.route('/predict', methods=['POST'])
@require_api_key
def predict():
    # Your code
    pass
```

### Rate Limiting
```python
from flask_limiter import Limiter

limiter = Limiter(app, key_func=lambda: request.remote_addr)

@app.route('/predict', methods=['POST'])
@limiter.limit("10 per minute")
def predict():
    # Your code
    pass
```

---

## 📈 Monitoring

### Add Logging
```python
import logging

logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

@app.route('/predict', methods=['POST'])
def predict():
    logging.info(f"Prediction request from {request.remote_addr}")
    # Your code
    logging.info(f"Prediction result: {result['diagnosis']}")
```

### Add Metrics
```python
prediction_count = 0
disease_counts = defaultdict(int)

@app.route('/predict', methods=['POST'])
def predict():
    global prediction_count
    prediction_count += 1
    disease_counts[result['diagnosis']] += 1
    # Your code

@app.route('/metrics')
def metrics():
    return jsonify({
        'total_predictions': prediction_count,
        'disease_distribution': dict(disease_counts)
    })
```

---

## ✅ Final Checklist

Before going live:

- [ ] All model files uploaded correctly
- [ ] Application builds successfully
- [ ] Test with multiple images
- [ ] Check mobile responsiveness
- [ ] Verify all diseases detect correctly
- [ ] Test with non-rice leaf images
- [ ] Check error handling
- [ ] Review security settings
- [ ] Add usage instructions
- [ ] Update README with your info
- [ ] Test API endpoints
- [ ] Check logs for errors
- [ ] Set up monitoring (optional)
- [ ] Share with friends for beta testing!

---

## 🆘 Getting Help

If you encounter issues:

1. **Check Logs**: Space → Logs tab
2. **Hugging Face Forum**: https://discuss.huggingface.co/
3. **GitHub Issues**: Create issue in your repo
4. **Discord**: Join Hugging Face Discord

---

## 🎉 You're Done!

Your Rice Leaf Disease Detection app is now live! 🚀

**Share your Space**:
```
https://huggingface.co/spaces/your-username/rice-leaf-disease-detection
```

**Example URLs**:
- App: `https://your-username-rice-leaf-disease-detection.hf.space`
- API: `https://your-username-rice-leaf-disease-detection.hf.space/predict`

---

**Happy Deploying! 🌾🤖**