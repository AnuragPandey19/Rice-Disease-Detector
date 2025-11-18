// Rice Leaf Disease Detection - Frontend JavaScript with Camera Support

// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const uploadContent = document.getElementById('uploadContent');
const imagePreview = document.getElementById('imagePreview');
const previewImage = document.getElementById('previewImage');
const removeImage = document.getElementById('removeImage');
const analyzeButton = document.getElementById('analyzeButton');
const loadingOverlay = document.getElementById('loadingOverlay');
const resultsSection = document.getElementById('resultsSection');
const closeResults = document.getElementById('closeResults');
const analyzeAnother = document.getElementById('analyzeAnother');

// Camera elements
const cameraButton = document.getElementById('cameraButton');
const cameraModal = document.getElementById('cameraModal');
const cameraVideo = document.getElementById('cameraVideo');
const cameraCanvas = document.getElementById('cameraCanvas');
const captureBtn = document.getElementById('captureBtn');
const switchCameraBtn = document.getElementById('switchCameraBtn');
const closeCameraBtn = document.getElementById('closeCameraBtn');
const cancelCameraBtn = document.getElementById('cancelCameraBtn');

// Result elements
const diagnosisBanner = document.getElementById('diagnosisBanner');
const diagnosisIcon = document.getElementById('diagnosisIcon');
const diagnosisTitle = document.getElementById('diagnosisTitle');
const confidenceValue = document.getElementById('confidenceValue');
const severityBadge = document.getElementById('severityBadge');
const descriptionText = document.getElementById('descriptionText');
const recommendationText = document.getElementById('recommendationText');
const stage1Pred = document.getElementById('stage1Pred');
const stage1Conf = document.getElementById('stage1Conf');
const stage2Used = document.getElementById('stage2Used');
const modelsUsed = document.getElementById('modelsUsed');

// State
let selectedFile = null;
let cameraStream = null;
let currentFacingMode = 'environment'; // Start with rear camera on mobile

// ============================================
// Event Listeners
// ============================================

// Click to upload
uploadArea.addEventListener('click', (e) => {
    // Don't trigger if clicking camera button
    if (e.target.closest('.camera-button')) {
        return;
    }
    if (!selectedFile) {
        fileInput.click();
    }
});

// File input change
fileInput.addEventListener('change', (e) => {
    handleFileSelect(e.target.files[0]);
});

// Drag and drop
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        handleFileSelect(file);
    } else {
        showError('Please upload a valid image file');
    }
});

// Remove image
removeImage.addEventListener('click', (e) => {
    e.stopPropagation();
    resetUpload();
});

// Analyze button
analyzeButton.addEventListener('click', () => {
    if (selectedFile) {
        analyzeImage();
    }
});

// Close results
closeResults.addEventListener('click', () => {
    resultsSection.style.display = 'none';
});

// Analyze another
analyzeAnother.addEventListener('click', () => {
    resetUpload();
    resultsSection.style.display = 'none';
    window.scrollTo({ top: 0, behavior: 'smooth' });
});

// Camera button
cameraButton.addEventListener('click', (e) => {
    e.stopPropagation();
    openCamera();
});

// Capture photo
captureBtn.addEventListener('click', () => {
    capturePhoto();
});

// Switch camera (front/back)
switchCameraBtn.addEventListener('click', () => {
    switchCamera();
});

// Close camera modal
closeCameraBtn.addEventListener('click', () => {
    closeCamera();
});

cancelCameraBtn.addEventListener('click', () => {
    closeCamera();
});

// Close camera modal when clicking outside
cameraModal.addEventListener('click', (e) => {
    if (e.target === cameraModal) {
        closeCamera();
    }
});

// Smooth scroll for nav links
document.querySelectorAll('.nav-link').forEach(link => {
    link.addEventListener('click', (e) => {
        e.preventDefault();
        const targetId = link.getAttribute('href').substring(1);
        const targetSection = document.getElementById(targetId);
        
        if (targetSection) {
            targetSection.scrollIntoView({ behavior: 'smooth' });
            
            // Update active link
            document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
            link.classList.add('active');
        }
    });
});

// ============================================
// Camera Functions
// ============================================

async function openCamera() {
    try {
        cameraModal.style.display = 'flex';
        
        // Request camera access
        const constraints = {
            video: {
                facingMode: currentFacingMode,
                width: { ideal: 1920 },
                height: { ideal: 1080 }
            },
            audio: false
        };
        
        cameraStream = await navigator.mediaDevices.getUserMedia(constraints);
        cameraVideo.srcObject = cameraStream;
        
    } catch (error) {
        console.error('Error accessing camera:', error);
        closeCamera();
        
        if (error.name === 'NotAllowedError') {
            showError('Camera access denied. Please allow camera access in your browser settings.');
        } else if (error.name === 'NotFoundError') {
            showError('No camera found on this device.');
        } else {
            showError('Unable to access camera. Please try again.');
        }
    }
}

async function switchCamera() {
    if (!cameraStream) return;
    
    // Toggle between front and back camera
    currentFacingMode = currentFacingMode === 'environment' ? 'user' : 'environment';
    
    // Stop current stream
    cameraStream.getTracks().forEach(track => track.stop());
    
    // Start new stream with different camera
    try {
        const constraints = {
            video: {
                facingMode: currentFacingMode,
                width: { ideal: 1920 },
                height: { ideal: 1080 }
            },
            audio: false
        };
        
        cameraStream = await navigator.mediaDevices.getUserMedia(constraints);
        cameraVideo.srcObject = cameraStream;
        
    } catch (error) {
        console.error('Error switching camera:', error);
        showError('Unable to switch camera. Your device may only have one camera.');
        
        // Revert to previous camera
        currentFacingMode = currentFacingMode === 'environment' ? 'user' : 'environment';
        openCamera();
    }
}

function capturePhoto() {
    if (!cameraStream) return;
    
    // Set canvas size to match video
    cameraCanvas.width = cameraVideo.videoWidth;
    cameraCanvas.height = cameraVideo.videoHeight;
    
    // Draw video frame to canvas
    const ctx = cameraCanvas.getContext('2d');
    ctx.drawImage(cameraVideo, 0, 0);
    
    // Convert canvas to blob
    cameraCanvas.toBlob((blob) => {
        if (blob) {
            // Create file from blob
            const file = new File([blob], 'camera-photo.jpg', { type: 'image/jpeg' });
            handleFileSelect(file);
            closeCamera();
        }
    }, 'image/jpeg', 0.95);
}

function closeCamera() {
    // Stop camera stream
    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
    }
    
    // Clear video
    cameraVideo.srcObject = null;
    
    // Hide modal
    cameraModal.style.display = 'none';
}

// ============================================
// File Handling Functions
// ============================================

function handleFileSelect(file) {
    if (!file) return;
    
    // Validate file type
    if (!file.type.startsWith('image/')) {
        showError('Please select an image file');
        return;
    }
    
    // Validate file size (10MB)
    if (file.size > 10 * 1024 * 1024) {
        showError('File size must be less than 10MB');
        return;
    }
    
    selectedFile = file;
    
    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        uploadContent.style.display = 'none';
        imagePreview.style.display = 'block';
        analyzeButton.disabled = false;
        
        // Add animation
        imagePreview.style.animation = 'fadeInUp 0.5s ease';
    };
    reader.readAsDataURL(file);
}

function resetUpload() {
    selectedFile = null;
    fileInput.value = '';
    uploadContent.style.display = 'block';
    imagePreview.style.display = 'none';
    previewImage.src = '';
    analyzeButton.disabled = true;
}

async function analyzeImage() {
    if (!selectedFile) return;
    
    // Show loading
    loadingOverlay.style.display = 'flex';
    
    try {
        // Create form data
        const formData = new FormData();
        formData.append('file', selectedFile);
        
        // Send request
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error('Analysis failed. Please try again.');
        }
        
        const result = await response.json();
        
        if (result.error) {
            throw new Error(result.error);
        }
        
        // Hide loading
        loadingOverlay.style.display = 'none';
        
        // Show results
        displayResults(result);
        
    } catch (error) {
        loadingOverlay.style.display = 'none';
        showError(error.message || 'An error occurred during analysis');
    }
}

function displayResults(result) {
    // Set diagnosis banner
    diagnosisIcon.textContent = result.icon;
    diagnosisTitle.textContent = result.diagnosis;
    confidenceValue.textContent = result.confidence;
    
    // Set banner color based on diagnosis
    diagnosisBanner.classList.remove('healthy', 'disease', 'non-leaf');
    if (result.diagnosis.includes('Healthy')) {
        diagnosisBanner.classList.add('healthy');
    } else if (result.diagnosis.includes('Not a Rice Leaf')) {
        diagnosisBanner.classList.add('non-leaf');
    } else {
        diagnosisBanner.classList.add('disease');
    }
    
    // Set severity
    severityBadge.textContent = result.severity;
    severityBadge.classList.remove('high', 'medium', 'low', 'none');
    
    const severityMap = {
        'High': 'high',
        'Medium': 'medium',
        'Low': 'low',
        'None': 'none',
        'N/A': 'none'
    };
    severityBadge.classList.add(severityMap[result.severity] || 'none');
    
    // Set description and recommendation
    descriptionText.textContent = result.description;
    recommendationText.textContent = result.recommendation;
    
    // Set technical details
    stage1Pred.textContent = result.details.stage1_prediction;
    stage1Conf.textContent = result.details.stage1_confidence;
    stage2Used.textContent = result.stage2_used ? 'Yes ✓' : 'No ✗';
    modelsUsed.textContent = result.details.models_used;
    
    // Show results section with animation
    resultsSection.style.display = 'block';
    resultsSection.style.animation = 'fadeInUp 0.5s ease';
    
    // Scroll to results
    setTimeout(() => {
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);
}

function showError(message) {
    // Create error notification
    const errorDiv = document.createElement('div');
    errorDiv.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: #ef4444;
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        z-index: 3000;
        animation: slideInRight 0.3s ease;
        max-width: 400px;
    `;
    errorDiv.textContent = message;
    
    document.body.appendChild(errorDiv);
    
    // Remove after 5 seconds
    setTimeout(() => {
        errorDiv.style.animation = 'slideOutRight 0.3s ease';
        setTimeout(() => errorDiv.remove(), 300);
    }, 5000);
}

// Add animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideInRight {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOutRight {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(100%);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);

// ============================================
// Initialize
// ============================================

console.log('🌾 Rice Leaf Disease Detection - Initialized');
console.log('Camera support:', navigator.mediaDevices ? 'Available' : 'Not available');
console.log('Ready to analyze images!');