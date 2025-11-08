// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const imageInput = document.getElementById('imageInput');
const browseBtn = document.getElementById('browseBtn');
const imagePreview = document.getElementById('imagePreview');
const previewImg = document.getElementById('previewImg');
const removeBtn = document.getElementById('removeBtn');
const predictBtn = document.getElementById('predictBtn');
const resultsSection = document.getElementById('resultsSection');
const loadingOverlay = document.getElementById('loadingOverlay');

let selectedFile = null;

// Initialize
document.addEventListener('DOMContentLoaded', function() {
    // Smooth scrolling for navigation
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        });
    });
    
    // Active nav link on scroll
    window.addEventListener('scroll', updateActiveNavLink);
});

function updateActiveNavLink() {
    const sections = document.querySelectorAll('section[id]');
    const navLinks = document.querySelectorAll('.nav-menu a');
    
    let current = '';
    sections.forEach(section => {
        const sectionTop = section.offsetTop;
        if (scrollY >= sectionTop - 100) {
            current = section.getAttribute('id');
        }
    });
    
    navLinks.forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('href') === `#${current}`) {
            link.classList.add('active');
        }
    });
}

// File Upload Handlers
browseBtn.addEventListener('click', () => {
    imageInput.click();
});

uploadArea.addEventListener('click', () => {
    imageInput.click();
});

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.style.borderColor = 'var(--secondary-color)';
    uploadArea.style.background = '#f0fdf4';
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.style.borderColor = 'var(--primary-color)';
    uploadArea.style.background = 'white';
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.style.borderColor = 'var(--primary-color)';
    uploadArea.style.background = 'white';
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFileSelect(files[0]);
    }
});

imageInput.addEventListener('change', (e) => {
    const files = e.target.files;
    if (files.length > 0) {
        handleFileSelect(files[0]);
    }
});

function handleFileSelect(file) {
    // Validate file type
    if (!file.type.startsWith('image/')) {
        showError('Please select an image file');
        return;
    }
    
    // Validate file size (max 16MB)
    if (file.size > 16 * 1024 * 1024) {
        showError('Image size should be less than 16MB');
        return;
    }
    
    selectedFile = file;
    
    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImg.src = e.target.result;
        uploadArea.style.display = 'none';
        imagePreview.style.display = 'block';
        predictBtn.style.display = 'block';
        resultsSection.style.display = 'none';
    };
    reader.readAsDataURL(file);
}

removeBtn.addEventListener('click', () => {
    selectedFile = null;
    imageInput.value = '';
    uploadArea.style.display = 'block';
    imagePreview.style.display = 'none';
    predictBtn.style.display = 'none';
    resultsSection.style.display = 'none';
});

// Prediction Handler
predictBtn.addEventListener('click', async () => {
    if (!selectedFile) {
        showError('Please select an image first');
        return;
    }
    
    // Show loading
    loadingOverlay.classList.add('active');
    
    // Prepare form data
    const formData = new FormData();
    formData.append('image', selectedFile);
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error('Prediction failed');
        }
        
        const data = await response.json();
        
        if (data.success) {
            displayResults(data);
        } else {
            showError(data.error || 'Prediction failed');
        }
    } catch (error) {
        console.error('Error:', error);
        showError('An error occurred. Please try again.');
    } finally {
        loadingOverlay.classList.remove('active');
    }
});

function displayResults(data) {
    const { prediction, disease_info } = data;
    
    // Show results section
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    
    // Display disease name and confidence
    document.getElementById('diseaseName').textContent = prediction.disease;
    document.getElementById('confidenceBadge').textContent = `${prediction.confidence}%`;
    
    // Display probabilities
    const probabilitiesDiv = document.getElementById('probabilities');
    probabilitiesDiv.innerHTML = '<h4 style="margin-bottom: 1rem;">📊 Probability Breakdown</h4>';
    
    for (const [disease, prob] of Object.entries(prediction.probabilities)) {
        const probItem = document.createElement('div');
        probItem.className = 'prob-item';
        probItem.innerHTML = `
            <span style="font-weight: 500;">${disease}</span>
            <div class="prob-bar">
                <div class="prob-fill" style="width: ${prob}%"></div>
            </div>
            <span style="font-weight: 600; color: var(--primary-color);">${prob}%</span>
        `;
        probabilitiesDiv.appendChild(probItem);
    }
    
    // Display disease information
    if (disease_info) {
        document.getElementById('diseaseDescription').textContent = disease_info.description || 'No description available';
        
        // Symptoms
        const symptomsList = document.getElementById('diseaseSymptoms');
        symptomsList.innerHTML = '';
        if (disease_info.symptoms) {
            disease_info.symptoms.forEach(symptom => {
                const li = document.createElement('li');
                li.textContent = symptom;
                symptomsList.appendChild(li);
            });
        }
        
        // Treatment
        const treatmentList = document.getElementById('diseaseTreatment');
        treatmentList.innerHTML = '';
        if (disease_info.treatment) {
            disease_info.treatment.forEach(treatment => {
                const li = document.createElement('li');
                li.textContent = treatment;
                treatmentList.appendChild(li);
            });
        }
        
        // Severity badge
        const severityBadge = document.getElementById('severityBadge');
        const severity = disease_info.severity || 'Unknown';
        severityBadge.textContent = `Severity: ${severity}`;
        
        // Set badge color based on severity
        if (severity === 'High') {
            severityBadge.style.background = 'var(--danger)';
            severityBadge.style.color = 'white';
        } else if (severity === 'Medium') {
            severityBadge.style.background = 'var(--warning)';
            severityBadge.style.color = 'white';
        } else {
            severityBadge.style.background = 'var(--success)';
            severityBadge.style.color = 'white';
        }
    }
}

function showError(message) {
    alert(message);
}

// Add animation on scroll
const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -100px 0px'
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.style.opacity = '1';
            entry.target.style.transform = 'translateY(0)';
        }
    });
}, observerOptions);

// Observe all feature cards and sections
document.querySelectorAll('.feature-card, .about-text').forEach(el => {
    el.style.opacity = '0';
    el.style.transform = 'translateY(30px)';
    el.style.transition = 'all 0.6s ease';
    observer.observe(el);
});