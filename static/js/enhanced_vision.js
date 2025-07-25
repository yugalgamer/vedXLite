/**
 * Enhanced Vision Assistant
 * =========================
 * JavaScript functionality for the enhanced Gemma3n vision processing
 */

import { speakText } from './feature_voice.js';

// Ensure critical functions are available globally immediately
// This prevents timing issues with module loading
window.startCameraCapture = null; // Placeholder, will be defined below

// Global variables
let currentImageFile = null;
let lastAnalysisResult = '';
let gemmaStatus = { available: false, enabled: false };
let cameraStream = null;
let isUsingCamera = false;
let capturedImageBlob = null;

// Initialize the enhanced vision system
function initEnhancedVision() {
    console.log('🔮 Initializing Enhanced Vision Assistant...');
    
    setupEventListeners();
    checkGemmaStatus();
    setupDragAndDrop();
    
    console.log('✅ Enhanced Vision Assistant initialized');
}

// Setup event listeners
function setupEventListeners() {
    // File input change
    const fileInput = document.getElementById('vision-file-input');
    if (fileInput) {
        fileInput.addEventListener('change', handleFileSelect);
    }
    
    // Upload area click (but not if clicking buttons)
    const uploadArea = document.getElementById('vision-upload-area');
    if (uploadArea) {
        uploadArea.addEventListener('click', (e) => {
            // Don't trigger file select if clicking on buttons
            if (!e.target.closest('.upload-btn') && !e.target.closest('.camera-btn')) {
                triggerFileSelect();
            }
        });
    }
    
    // Camera button click
    const cameraBtn = document.getElementById('vision-camera-btn');
    if (cameraBtn) {
        cameraBtn.addEventListener('click', startCameraCapture);
    }
    
    // Upload button click
    const uploadBtn = document.getElementById('vision-upload-btn');
    if (uploadBtn) {
        uploadBtn.addEventListener('click', triggerFileSelect);
    }
    
    // Analyze button click
    const analyzeBtn = document.getElementById('analyze-vision-btn');
    if (analyzeBtn) {
        analyzeBtn.addEventListener('click', analyzeVisionImage);
    }
    
    // Remove button click
    const removeBtn = document.getElementById('remove-vision-btn');
    if (removeBtn) {
        removeBtn.addEventListener('click', removeVisionImage);
    }
    
    // Quick action button clicks
    const speakBtn = document.getElementById('speak-results-btn');
    if (speakBtn) {
        speakBtn.addEventListener('click', speakResults);
    }
    
    const copyBtn = document.getElementById('copy-results-btn');
    if (copyBtn) {
        copyBtn.addEventListener('click', copyResults);
    }
    
    const analyzeAgainBtn = document.getElementById('analyze-again-btn');
    if (analyzeAgainBtn) {
        analyzeAgainBtn.addEventListener('click', analyzeAgain);
    }
    
    // Gemma toggle button
    const gemmaToggleBtn = document.getElementById('gemma-toggle-btn');
    if (gemmaToggleBtn) {
        gemmaToggleBtn.addEventListener('click', toggleGemmaReasoning);
    }
    
    // Question input enter key
    const questionInput = document.getElementById('vision-question-input');
    if (questionInput) {
        questionInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && e.ctrlKey) {
                analyzeVisionImage();
            }
        });
    }
}

// Setup drag and drop functionality
function setupDragAndDrop() {
    const uploadArea = document.getElementById('vision-upload-area');
    if (!uploadArea) return;
    
    // Prevent default drag behaviors
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });
    
    // Highlight drop area when item is dragged over it
    ['dragenter', 'dragover'].forEach(eventName => {
        uploadArea.addEventListener(eventName, highlight, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, unhighlight, false);
    });
    
    // Handle dropped files
    uploadArea.addEventListener('drop', handleDrop, false);
}

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

function highlight(e) {
    const uploadArea = document.getElementById('vision-upload-area');
    uploadArea.classList.add('drag-over');
}

function unhighlight(e) {
    const uploadArea = document.getElementById('vision-upload-area');
    uploadArea.classList.remove('drag-over');
}

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    
    if (files.length > 0) {
        const file = files[0];
        if (file.type.startsWith('image/')) {
            handleFileSelect({ target: { files: [file] } });
        } else {
            showNotification('Please drop an image file', 'error');
        }
    }
}

// Trigger file selection
window.triggerVisionUpload = function() {
    const fileInput = document.getElementById('vision-file-input');
    if (fileInput) {
        fileInput.click();
    }
};

// Trigger file selection (alias for compatibility)
function triggerFileSelect() {
    const fileInput = document.getElementById('vision-file-input');
    if (fileInput) {
        fileInput.click();
    }
}

// Handle file selection
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    // Validate file type
    if (!file.type.startsWith('image/')) {
        showNotification('Please select an image file', 'error');
        return;
    }
    
    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
        showNotification('Image file is too large. Please select a file under 10MB', 'error');
        return;
    }
    
    currentImageFile = file;
    displayImagePreview(file);
}

// Display image preview
function displayImagePreview(file) {
    const uploadPlaceholder = document.getElementById('vision-upload-placeholder');
    const imagePreview = document.getElementById('vision-image-preview');
    const previewImg = document.getElementById('vision-preview-img');
    
    if (!uploadPlaceholder || !imagePreview || !previewImg) return;
    
    // Create object URL for preview
    const objectURL = URL.createObjectURL(file);
    previewImg.src = objectURL;
    
    // Show preview, hide placeholder
    uploadPlaceholder.style.display = 'none';
    imagePreview.style.display = 'block';
    
    // Update results area
    updateResultsWelcome('Image loaded! Click "Analyze Image" or ask a question to get started.');
}

// Remove current image
window.removeVisionImage = function() {
    const uploadPlaceholder = document.getElementById('vision-upload-placeholder');
    const imagePreview = document.getElementById('vision-image-preview');
    const previewImg = document.getElementById('vision-preview-img');
    const fileInput = document.getElementById('vision-file-input');
    
    // Reset file input
    if (fileInput) fileInput.value = '';
    
    // Reset preview
    if (previewImg) {
        URL.revokeObjectURL(previewImg.src);
        previewImg.src = '';
    }
    
    // Show placeholder, hide preview
    if (uploadPlaceholder) uploadPlaceholder.style.display = 'flex';
    if (imagePreview) imagePreview.style.display = 'none';
    
    // Reset variables
    currentImageFile = null;
    lastAnalysisResult = '';
    
    // Reset results area
    updateResultsWelcome();
    hideQuickActions();
};

// Analyze the current image
window.analyzeVisionImage = async function() {
    if (!currentImageFile) {
        showNotification('Please select an image first', 'error');
        return;
    }
    
    const questionInput = document.getElementById('vision-question-input');
    const userQuestion = questionInput ? questionInput.value.trim() : '';
    
    // Show processing overlay
    showProcessingOverlay('Analyzing image with enhanced AI...');
    
    try {
        // Prepare form data
        const formData = new FormData();
        formData.append('image', currentImageFile);
        formData.append('prompt', userQuestion || 'Describe this image for a blind person, focusing on important objects, their locations, and any potential hazards or useful information.');
        
        // Add user context if available
        if (window.currentUserName) formData.append('username', window.currentUserName);
        if (window.currentUserRole) formData.append('role', window.currentUserRole);
        
        // Use enhanced vision endpoint if available
        const endpoint = gemmaStatus.available ? '/api/enhanced-vision' : '/api/analyze';
        
        updateProcessingStatus('Sending image to AI analysis...');
        
        const response = await fetch(endpoint, {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success !== false && (data.response || data.analysis)) {
            const result = data.response || data.analysis;
            lastAnalysisResult = result;
            
            updateProcessingStatus('Formatting results...');
            
            // Display results
            displayAnalysisResult(result, data);
            
            // Show quick actions
            showQuickActions();
            
            showNotification('Analysis completed successfully!', 'success');
        } else {
            throw new Error(data.error || 'Analysis failed');
        }
    } catch (error) {
        console.error('Vision analysis error:', error);
        displayError(`Analysis failed: ${error.message}`);
        showNotification('Analysis failed. Please try again.', 'error');
    } finally {
        hideProcessingOverlay();
    }
};

// Display analysis result
function displayAnalysisResult(result, metadata = {}) {
    const resultsContent = document.getElementById('vision-results-content');
    const modeIndicator = document.getElementById('mode-indicator');
    
    if (!resultsContent) return;
    
    // Update processing mode indicator
    if (modeIndicator) {
        if (metadata.enhanced_processing) {
            modeIndicator.textContent = 'Enhanced Mode';
            modeIndicator.className = 'mode-indicator';
        } else {
            modeIndicator.textContent = 'Basic Mode';
            modeIndicator.className = 'mode-indicator basic';
        }
    }
    
    // Create result HTML
    const resultHTML = `
        <div class="analysis-result">
            <h4><i class="fas fa-eye"></i> Vision Analysis Result</h4>
            <div class="result-text">${formatResultText(result)}</div>
            ${metadata.vision_description ? `
                <div class="basic-description" style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid var(--border-color);">
                    <h5 style="color: var(--secondary-text); margin-bottom: 0.5rem;">Raw Vision Detection:</h5>
                    <p style="color: var(--secondary-text); font-size: 0.9rem;">${metadata.vision_description}</p>
                </div>
            ` : ''}
            ${metadata.metadata ? `
                <div class="analysis-metadata" style="margin-top: 1rem; padding: 1rem; background: var(--accent-bg); border-radius: var(--border-radius-sm);">
                    <small style="color: var(--secondary-text);">
                        Processing time: ${(metadata.metadata.processing_time || 0).toFixed(2)}s | 
                        Source: ${metadata.metadata.source || 'unknown'} | 
                        Template: ${metadata.metadata.template_type || 'default'}
                    </small>
                </div>
            ` : ''}
        </div>
    `;
    
    resultsContent.innerHTML = resultHTML;
}

// Format result text with basic markdown support
function formatResultText(text) {
    if (!text) return '';
    
    // Convert line breaks
    text = text.replace(/\n/g, '<br>');
    
    // Basic markdown formatting (same as chat)
    text = text.replace(/\*\*\*(.*?)\*\*\*/g, '<strong><em>$1</em></strong>');
    text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    text = text.replace(/\*(.*?)\*/g, '<em>$1</em>');
    
    return text;
}

// Display error message
function displayError(message) {
    const resultsContent = document.getElementById('vision-results-content');
    if (!resultsContent) return;
    
    resultsContent.innerHTML = `
        <div class="error-message" style="text-align: center; color: var(--danger-color); padding: 2rem;">
            <i class="fas fa-exclamation-triangle" style="font-size: 3rem; margin-bottom: 1rem; display: block;"></i>
            <h4>Analysis Error</h4>
            <p>${message}</p>
            <button onclick="analyzeAgain()" class="quick-action-btn" style="margin-top: 1rem;">
                <i class="fas fa-redo"></i> Try Again
            </button>
        </div>
    `;
}

// Update results welcome message
function updateResultsWelcome(message = null) {
    const resultsContent = document.getElementById('vision-results-content');
    if (!resultsContent) return;
    
    if (message) {
        resultsContent.innerHTML = `
            <div class="welcome-message">
                <i class="fas fa-info-circle"></i>
                <p>${message}</p>
            </div>
        `;
    } else {
        resultsContent.innerHTML = `
            <div class="welcome-message">
                <i class="fas fa-info-circle"></i>
                <p>Upload an image to get started with enhanced AI-powered vision analysis specifically designed for blind users.</p>
                <div class="features-list">
                    <h4>Features:</h4>
                    <ul>
                        <li><i class="fas fa-check"></i> Detailed object detection and positioning</li>
                        <li><i class="fas fa-check"></i> Safety hazard identification</li>
                        <li><i class="fas fa-check"></i> Navigation assistance</li>
                        <li><i class="fas fa-check"></i> Context-aware responses</li>
                        <li><i class="fas fa-check"></i> Voice-optimized descriptions</li>
                    </ul>
                </div>
            </div>
        `;
    }
}

// Quick action functions
window.speakResults = function() {
    if (!lastAnalysisResult) {
        showNotification('No results to speak', 'error');
        return;
    }
    
    // Clean text for speech (remove HTML and markdown)
    const cleanText = lastAnalysisResult
        .replace(/<[^>]*>/g, '') // Remove HTML tags
        .replace(/\*\*\*(.*?)\*\*\*/g, '$1') // Remove bold italic
        .replace(/\*\*(.*?)\*\*/g, '$1') // Remove bold
        .replace(/\*(.*?)\*/g, '$1'); // Remove italic
    
    speakText(cleanText);
    showNotification('Speaking results...', 'info');
};

window.copyResults = function() {
    if (!lastAnalysisResult) {
        showNotification('No results to copy', 'error');
        return;
    }
    
    // Clean text for copying
    const cleanText = lastAnalysisResult
        .replace(/<[^>]*>/g, '') // Remove HTML tags
        .replace(/\*\*\*(.*?)\*\*\*/g, '$1') // Remove bold italic
        .replace(/\*\*(.*?)\*\*/g, '$1') // Remove bold
        .replace(/\*(.*?)\*/g, '$1'); // Remove italic
    
    navigator.clipboard.writeText(cleanText).then(() => {
        showNotification('Results copied to clipboard!', 'success');
    }).catch(() => {
        showNotification('Failed to copy to clipboard', 'error');
    });
};

window.analyzeAgain = function() {
    window.analyzeVisionImage();
};

// Camera capture functions
// Define the function first
async function startCameraCapture() {
    try {
        // Check if browser supports camera access
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            showNotification('Camera access is not supported in your browser', 'error');
            return;
        }

        // Request camera access
        cameraStream = await navigator.mediaDevices.getUserMedia({ 
            video: { 
                width: { ideal: 1920 },
                height: { ideal: 1080 },
                facingMode: 'environment' // Use back camera if available
            } 
        });

        // Show camera preview
        showCameraPreview();
        isUsingCamera = true;
        
        showNotification('Camera started! Position your subject and click "Capture Image"', 'success');
    } catch (error) {
        console.error('Camera access error:', error);
        if (error.name === 'NotAllowedError') {
            showNotification('Camera access denied. Please allow camera permissions and try again.', 'error');
        } else if (error.name === 'NotFoundError') {
            showNotification('No camera found on this device', 'error');
        } else {
            showNotification(`Camera error: ${error.message}`, 'error');
        }
    }
};

function showCameraPreview() {
    const uploadPlaceholder = document.getElementById('vision-upload-placeholder');
    const imagePreview = document.getElementById('vision-image-preview');
    
    if (!uploadPlaceholder || !imagePreview) return;

    // Create camera preview container
    const cameraContainer = document.createElement('div');
    cameraContainer.id = 'camera-preview-container';
    cameraContainer.style.cssText = `
        position: relative;
        width: 100%;
        height: 400px;
        background: #000;
        border-radius: var(--border-radius);
        overflow: hidden;
        display: flex;
        align-items: center;
        justify-content: center;
    `;

    // Create video element for live preview
    const video = document.createElement('video');
    video.id = 'camera-video';
    video.autoplay = true;
    video.muted = true;
    video.playsInline = true;
    video.style.cssText = `
        width: 100%;
        height: 100%;
        object-fit: cover;
    `;
    video.srcObject = cameraStream;

    // Create camera controls overlay
    const controls = document.createElement('div');
    controls.className = 'camera-controls';
    controls.style.cssText = `
        position: absolute;
        bottom: 20px;
        left: 50%;
        transform: translateX(-50%);
        display: flex;
        gap: 15px;
        z-index: 10;
    `;

    // Capture button
    const captureBtn = document.createElement('button');
    captureBtn.className = 'camera-btn capture-btn';
    captureBtn.innerHTML = '<i class="fas fa-camera"></i> Capture Image';
    captureBtn.style.cssText = `
        background: var(--accent-color);
        color: white;
        border: none;
        padding: 12px 20px;
        border-radius: var(--border-radius);
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(88, 166, 255, 0.3);
    `;
    captureBtn.onclick = captureImage;

    // Stop camera button
    const stopBtn = document.createElement('button');
    stopBtn.className = 'camera-btn stop-btn';
    stopBtn.innerHTML = '<i class="fas fa-times"></i> Stop Camera';
    stopBtn.style.cssText = `
        background: var(--danger-color);
        color: white;
        border: none;
        padding: 12px 20px;
        border-radius: var(--border-radius);
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
    `;
    stopBtn.onclick = stopCamera;

    controls.appendChild(captureBtn);
    controls.appendChild(stopBtn);
    cameraContainer.appendChild(video);
    cameraContainer.appendChild(controls);

    // Replace upload placeholder with camera preview
    uploadPlaceholder.style.display = 'none';
    imagePreview.style.display = 'none';
    
    // Insert camera container
    const uploadArea = document.getElementById('vision-upload-area');
    uploadArea.appendChild(cameraContainer);

    // Update results area
    updateResultsWelcome('Camera is live! Position your subject and click "Capture Image" when ready.');
}

function captureImage() {
    const video = document.getElementById('camera-video');
    if (!video) return;

    // Create canvas to capture frame
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Draw current video frame to canvas
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Convert to blob
    canvas.toBlob((blob) => {
        if (blob) {
            capturedImageBlob = blob;
            
            // Create file from blob
            const file = new File([blob], 'captured-image.jpg', { type: 'image/jpeg' });
            currentImageFile = file;
            
            // Show preview with confirmation
            showCapturePreview(canvas.toDataURL());
        }
    }, 'image/jpeg', 0.8);
}

function showCapturePreview(dataURL) {
    const uploadArea = document.getElementById('vision-upload-area');
    const cameraContainer = document.getElementById('camera-preview-container');
    
    if (cameraContainer) {
        cameraContainer.remove();
    }

    // Create preview container
    const previewContainer = document.createElement('div');
    previewContainer.id = 'capture-preview-container';
    previewContainer.style.cssText = `
        position: relative;
        width: 100%;
        text-align: center;
        background: var(--surface-color);
        border-radius: var(--border-radius);
        padding: 20px;
    `;

    // Preview image
    const previewImg = document.createElement('img');
    previewImg.src = dataURL;
    previewImg.style.cssText = `
        max-width: 100%;
        height: auto;
        border-radius: var(--border-radius);
        box-shadow: var(--shadow-lg);
        margin-bottom: 20px;
    `;

    // Confirmation message
    const message = document.createElement('h4');
    message.textContent = 'Are you satisfied with this image?';
    message.style.cssText = `
        color: var(--text-color);
        margin-bottom: 20px;
    `;

    // Action buttons
    const buttonContainer = document.createElement('div');
    buttonContainer.style.cssText = `
        display: flex;
        gap: 15px;
        justify-content: center;
        flex-wrap: wrap;
    `;

    // Use this image button
    const useBtn = document.createElement('button');
    useBtn.className = 'action-btn success-btn';
    useBtn.innerHTML = '<i class="fas fa-check"></i> Use This Image';
    useBtn.style.cssText = `
        background: var(--success-color);
        color: white;
        border: none;
        padding: 12px 20px;
        border-radius: var(--border-radius);
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
    `;
    useBtn.onclick = confirmCapturedImage;

    // Retake button
    const retakeBtn = document.createElement('button');
    retakeBtn.className = 'action-btn warning-btn';
    retakeBtn.innerHTML = '<i class="fas fa-redo"></i> Retake Photo';
    retakeBtn.style.cssText = `
        background: var(--warning-color);
        color: white;
        border: none;
        padding: 12px 20px;
        border-radius: var(--border-radius);
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
    `;
    retakeBtn.onclick = () => {
        previewContainer.remove();
        showCameraPreview();
    };

    // Cancel button
    const cancelBtn = document.createElement('button');
    cancelBtn.className = 'action-btn danger-btn';
    cancelBtn.innerHTML = '<i class="fas fa-times"></i> Cancel';
    cancelBtn.style.cssText = `
        background: var(--danger-color);
        color: white;
        border: none;
        padding: 12px 20px;
        border-radius: var(--border-radius);
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
    `;
    cancelBtn.onclick = () => {
        stopCamera();
        window.removeVisionImage();
    };

    buttonContainer.appendChild(useBtn);
    buttonContainer.appendChild(retakeBtn);
    buttonContainer.appendChild(cancelBtn);

    previewContainer.appendChild(previewImg);
    previewContainer.appendChild(message);
    previewContainer.appendChild(buttonContainer);

    uploadArea.appendChild(previewContainer);

    // Update results area
    updateResultsWelcome('Image captured! Review the photo and choose an action below.');
}

function confirmCapturedImage() {
    const previewContainer = document.getElementById('capture-preview-container');
    if (previewContainer) {
        previewContainer.remove();
    }

    // Stop camera stream
    stopCamera();

    // Show normal image preview
    displayImagePreview(currentImageFile);
    
    showNotification('Image ready for analysis! Click "Analyze Image" or ask a question.', 'success');
}

function stopCamera() {
    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
    }
    
    isUsingCamera = false;
    
    // Remove camera preview if it exists
    const cameraContainer = document.getElementById('camera-preview-container');
    if (cameraContainer) {
        cameraContainer.remove();
    }
    
    // Show upload placeholder again if no image
    if (!currentImageFile) {
        const uploadPlaceholder = document.getElementById('vision-upload-placeholder');
        if (uploadPlaceholder) {
            uploadPlaceholder.style.display = 'flex';
        }
        updateResultsWelcome();
    }
    
    showNotification('Camera stopped', 'info');
}

// Show/hide quick actions
function showQuickActions() {
    const quickActions = document.getElementById('vision-quick-actions');
    if (quickActions) {
        quickActions.style.display = 'flex';
    }
}

function hideQuickActions() {
    const quickActions = document.getElementById('vision-quick-actions');
    if (quickActions) {
        quickActions.style.display = 'none';
    }
}

// Processing overlay functions
function showProcessingOverlay(message = 'Processing...') {
    const overlay = document.getElementById('vision-processing-overlay');
    const statusText = document.getElementById('processing-status');
    
    if (overlay) {
        overlay.style.display = 'flex';
    }
    
    if (statusText) {
        statusText.textContent = message;
    }
}

function hideProcessingOverlay() {
    const overlay = document.getElementById('vision-processing-overlay');
    if (overlay) {
        overlay.style.display = 'none';
    }
}

function updateProcessingStatus(message) {
    const statusText = document.getElementById('processing-status');
    if (statusText) {
        statusText.textContent = message;
    }
}

// Gemma status functions
async function checkGemmaStatus() {
    try {
        const response = await fetch('/api/gemma-status');
        const data = await response.json();
        
        gemmaStatus = {
            available: data.available || false,
            enabled: data.enabled || false,
            model_name: data.model_name || 'gemma3n:latest',
            system_status: data.system_status || {}
        };
        
        updateStatusDisplay();
    } catch (error) {
        console.error('Failed to check Gemma status:', error);
        gemmaStatus = { available: false, enabled: false };
        updateStatusDisplay();
    }
}

function updateStatusDisplay() {
    const statusIcon = document.getElementById('gemma-status-icon');
    const statusText = document.getElementById('gemma-status-text');
    const toggleBtn = document.getElementById('gemma-toggle-btn');
    const toggleText = document.getElementById('toggle-text');
    
    if (statusIcon && statusText) {
        if (gemmaStatus.available && gemmaStatus.enabled) {
            statusIcon.className = 'fas fa-brain status-icon';
            statusIcon.style.color = 'var(--success-color)';
            statusText.textContent = `Enhanced Gemma3n reasoning active (${gemmaStatus.model_name})`;
        } else if (gemmaStatus.available) {
            statusIcon.className = 'fas fa-brain status-icon';
            statusIcon.style.color = 'var(--warning-color)';
            statusText.textContent = 'Gemma3n available but disabled - using basic vision';
        } else {
            statusIcon.className = 'fas fa-exclamation-triangle status-icon';
            statusIcon.style.color = 'var(--danger-color)';
            statusText.textContent = 'Enhanced reasoning unavailable - using basic vision';
        }
    }
    
    if (toggleBtn && toggleText) {
        if (!gemmaStatus.available) {
            toggleBtn.classList.add('disabled');
            toggleBtn.disabled = true;
            toggleText.textContent = 'Not Available';
        } else {
            toggleBtn.classList.remove('disabled');
            toggleBtn.disabled = false;
            toggleText.textContent = gemmaStatus.enabled ? 'Disable Enhanced Mode' : 'Enable Enhanced Mode';
        }
    }
}

// Toggle Gemma reasoning
window.toggleGemmaReasoning = async function() {
    if (!gemmaStatus.available) {
        showNotification('Enhanced reasoning is not available', 'error');
        return;
    }
    
    try {
        const response = await fetch('/api/gemma-toggle', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ enable: !gemmaStatus.enabled })
        });
        
        const data = await response.json();
        
        if (data.success) {
            gemmaStatus.enabled = data.enabled;
            updateStatusDisplay();
            showNotification(data.message, 'success');
        } else {
            throw new Error(data.error || 'Toggle failed');
        }
    } catch (error) {
        console.error('Failed to toggle Gemma:', error);
        showNotification(`Failed to toggle enhanced mode: ${error.message}`, 'error');
    }
};

// Utility functions
function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: ${type === 'success' ? 'var(--success-color)' : 
                    type === 'error' ? 'var(--danger-color)' : 
                    type === 'warning' ? 'var(--warning-color)' : 
                    'var(--accent-color)'};
        color: white;
        padding: 1rem 1.5rem;
        border-radius: var(--border-radius);
        box-shadow: var(--shadow-lg);
        z-index: 2000;
        font-weight: 600;
        animation: slideInRight 0.3s ease-out;
        max-width: 300px;
        word-wrap: break-word;
    `;
    notification.textContent = message;
    
    document.body.appendChild(notification);
    
    // Auto remove after 4 seconds
    setTimeout(() => {
        notification.style.animation = 'slideOutRight 0.3s ease-in';
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 300);
    }, 4000);
}

// Add CSS animations for notifications
if (!document.getElementById('notification-styles')) {
    const style = document.createElement('style');
    style.id = 'notification-styles';
    style.textContent = `
        @keyframes slideInRight {
            from { transform: translateX(100%); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
        @keyframes slideOutRight {
            from { transform: translateX(0); opacity: 1; }
            to { transform: translateX(100%); opacity: 0; }
        }
        .upload-area.drag-over {
            background: rgba(88, 166, 255, 0.1) !important;
            border-color: var(--accent-hover) !important;
            transform: scale(1.02) !important;
        }
    `;
    document.head.appendChild(style);
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Only initialize if we're on a page with the enhanced vision tab
    if (document.getElementById('enhanced-vision-tab')) {
        console.log('🔮 Enhanced Vision tab found, initializing...');
        initEnhancedVision();
        
        // Debug: Check if critical elements exist
        const analyzeBtn = document.getElementById('analyze-vision-btn');
        const uploadBtn = document.getElementById('vision-upload-btn');
        const cameraBtn = document.getElementById('vision-camera-btn');
        
        console.log('Enhanced Vision Debug:', {
            analyzeBtn: !!analyzeBtn,
            uploadBtn: !!uploadBtn,
            cameraBtn: !!cameraBtn,
            startCameraCapture: typeof window.startCameraCapture
        });
    } else {
        console.log('❌ Enhanced Vision tab not found');
    }
});

// Make critical functions available globally
window.startCameraCapture = startCameraCapture;

// Add debug logging
console.log('Enhanced Vision: startCameraCapture function assigned to window:', typeof window.startCameraCapture);

// Export functions for use in other modules
export { 
    initEnhancedVision, 
    checkGemmaStatus, 
    updateStatusDisplay 
};
