/**
 * Face Insight - AI Face Analyzer
 * Frontend JavaScript for handling image uploads and displaying results
 */

// Get DOM elements
const uploadBox = document.getElementById('uploadBox');
const imageInput = document.getElementById('imageInput');
const loading = document.getElementById('loading');
const errorMessage = document.getElementById('errorMessage');
const resultsSection = document.getElementById('resultsSection');
const resultCanvas = document.getElementById('resultCanvas');
const facesContainer = document.getElementById('facesContainer');

// Original image for drawing
let originalImage = null;

/**
 * Initialize event listeners
 */
function init() {
    // Click to upload
    uploadBox.addEventListener('click', () => {
        imageInput.click();
    });

    // File input change
    imageInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            handleFile(e.target.files[0]);
        }
    });

    // Drag and drop handlers
    uploadBox.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadBox.classList.add('dragover');
    });

    uploadBox.addEventListener('dragleave', () => {
        uploadBox.classList.remove('dragover');
    });

    uploadBox.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadBox.classList.remove('dragover');
        
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleFile(e.dataTransfer.files[0]);
        }
    });
}

/**
 * Handle file selection/upload
 * @param {File} file - The image file to process
 */
function handleFile(file) {
    // Validate file type
    if (!file.type.startsWith('image/')) {
        showError('Please select a valid image file.');
        return;
    }

    // Validate file size (16MB max)
    if (file.size > 16 * 1024 * 1024) {
        showError('File size too large. Please upload an image smaller than 16MB.');
        return;
    }

    // Hide previous results and errors
    hideError();
    hideResults();

    // Show loading
    showLoading();

    // Create FormData for file upload
    const formData = new FormData();
    formData.append('image', file);

    // Send request to backend
    fetch('/analyze', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        hideLoading();
        
        if (data.success) {
            // Load image and display results
            loadImageAndDisplayResults(file, data.faces);
        } else {
            showError(data.message || data.error || 'Failed to analyze image.');
        }
    })
    .catch(error => {
        hideLoading();
        showError('Error connecting to server: ' + error.message);
        console.error('Error:', error);
    });
}

/**
 * Load image and display analysis results
 * @param {File} file - The image file
 * @param {Array} faces - Array of face analysis results
 */
function loadImageAndDisplayResults(file, faces) {
    const reader = new FileReader();
    
    reader.onload = (e) => {
        const img = new Image();
        img.onload = () => {
            originalImage = img;
            
            // Draw image with bounding boxes
            drawImageWithBoxes(img, faces);
            
            // Display face analysis cards
            displayFaceCards(faces);
            
            // Show results section
            showResults();
        };
        img.src = e.target.result;
    };
    
    reader.readAsDataURL(file);
}

/**
 * Draw image with bounding boxes around detected faces
 * @param {Image} img - The image to draw
 * @param {Array} faces - Array of face data with bounding boxes
 */
function drawImageWithBoxes(img, faces) {
    const canvas = resultCanvas;
    const ctx = canvas.getContext('2d');
    
    // Set canvas size to match image
    canvas.width = img.width;
    canvas.height = img.height;
    
    // Draw the image
    ctx.drawImage(img, 0, 0);
    
    // Draw bounding boxes for each face
    faces.forEach((face, index) => {
        const box = face.bounding_box;
        
        // Draw rectangle
        ctx.strokeStyle = '#667eea';
        ctx.lineWidth = 3;
        ctx.strokeRect(box.x, box.y, box.width, box.height);
        
        // Draw label background
        const labelText = `Face ${index + 1}`;
        ctx.font = 'bold 16px Arial';
        const textMetrics = ctx.measureText(labelText);
        const labelWidth = textMetrics.width + 10;
        const labelHeight = 25;
        
        ctx.fillStyle = '#667eea';
        ctx.fillRect(box.x, box.y - labelHeight, labelWidth, labelHeight);
        
        // Draw label text
        ctx.fillStyle = 'white';
        ctx.fillText(labelText, box.x + 5, box.y - 8);
    });
}

/**
 * Display face analysis cards for each detected face
 * @param {Array} faces - Array of face analysis results
 */
function displayFaceCards(faces) {
    facesContainer.innerHTML = '';
    
    faces.forEach((face, index) => {
        const card = createFaceCard(face, index + 1);
        facesContainer.appendChild(card);
    });
}

/**
 * Create a face analysis card element
 * @param {Object} face - Face analysis data
 * @param {number} faceNumber - Face number (1, 2, 3, etc.)
 * @returns {HTMLElement} The card element
 */
function createFaceCard(face, faceNumber) {
    const card = document.createElement('div');
    card.className = 'face-card';
    
    // Format emotion with emoji
    const emotionEmoji = getEmotionEmoji(face.emotion);
    
    // Format gender
    const genderDisplay = face.gender === 'Man' ? '👨 Male' : face.gender === 'Woman' ? '👩 Female' : face.gender;
    
    // Symmetry explanation
    const symmetryLevel = getSymmetryLevel(face.symmetry);
    
    card.innerHTML = `
        <h3>Face ${faceNumber}</h3>
        
        <div class="analysis-item">
            <label>Age</label>
            <div class="value">${face.age} years</div>
        </div>
        
        <div class="analysis-item">
            <label>Gender</label>
            <div class="value">${genderDisplay}</div>
            ${face.confidence.gender > 0 ? `<div class="confidence">Confidence: ${face.confidence.gender}%</div>` : ''}
        </div>
        
        <div class="analysis-item">
            <label>Emotion</label>
            <div class="value">${emotionEmoji} ${face.emotion}</div>
            ${face.confidence.emotion > 0 ? `<div class="confidence">Confidence: ${face.confidence.emotion}%</div>` : ''}
        </div>
        
        <div class="analysis-item">
            <label>Facial Symmetry</label>
            <div class="value">${face.symmetry}%</div>
            <div class="symmetry-bar">
                <div class="symmetry-fill" style="width: ${face.symmetry}%">
                    ${face.symmetry}%
                </div>
            </div>
            <div class="symmetry-explanation">
                ${symmetryLevel}
            </div>
        </div>
    `;
    
    return card;
}

/**
 * Get emoji for emotion
 * @param {string} emotion - Emotion name
 * @returns {string} Emoji character
 */
function getEmotionEmoji(emotion) {
    const emojiMap = {
        'happy': '😊',
        'sad': '😢',
        'angry': '😠',
        'surprise': '😲',
        'fear': '😨',
        'disgust': '🤢',
        'neutral': '😐'
    };
    
    return emojiMap[emotion.toLowerCase()] || '😐';
}

/**
 * Get symmetry level description
 * @param {number} symmetry - Symmetry percentage
 * @returns {string} Description text
 */
function getSymmetryLevel(symmetry) {
    if (symmetry >= 90) {
        return 'Excellent symmetry - Very balanced facial features';
    } else if (symmetry >= 75) {
        return 'Good symmetry - Well-proportioned face';
    } else if (symmetry >= 60) {
        return 'Moderate symmetry - Slight asymmetry present';
    } else {
        return 'Lower symmetry - Noticeable asymmetry detected';
    }
}

/**
 * Show loading indicator
 */
function showLoading() {
    loading.style.display = 'block';
    uploadBox.style.pointerEvents = 'none';
    uploadBox.style.opacity = '0.6';
}

/**
 * Hide loading indicator
 */
function hideLoading() {
    loading.style.display = 'none';
    uploadBox.style.pointerEvents = 'auto';
    uploadBox.style.opacity = '1';
}

/**
 * Show error message
 * @param {string} message - Error message to display
 */
function showError(message) {
    errorMessage.textContent = message;
    errorMessage.style.display = 'block';
}

/**
 * Hide error message
 */
function hideError() {
    errorMessage.style.display = 'none';
}

/**
 * Show results section
 */
function showResults() {
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

/**
 * Hide results section
 */
function hideResults() {
    resultsSection.style.display = 'none';
}

/**
 * Try Again - Reset everything and allow new upload
 */
function tryAgain() {
    // Hide results
    hideResults();
    
    // Hide any error messages
    hideError();
    
    // Clear the canvas
    const ctx = resultCanvas.getContext('2d');
    ctx.clearRect(0, 0, resultCanvas.width, resultCanvas.height);
    
    // Clear face cards
    facesContainer.innerHTML = '';
    
    // Reset file input
    imageInput.value = '';
    
    // Reset original image
    originalImage = null;
    
    // Scroll to top smoothly
    window.scrollTo({
        top: 0,
        behavior: 'smooth'
    });
    
    // Re-enable upload box
    uploadBox.style.pointerEvents = 'auto';
    uploadBox.style.opacity = '1';
}

// Initialize when page loads
document.addEventListener('DOMContentLoaded', init);

