// Yoga Pose Detection Web App - Frontend JavaScript

let video = null;
let canvas = null;
let ctx = null;
let isStreaming = false;
let sessionId = null;
let animationFrameId = null;
let fps = 30;
let frameCount = 0;
let lastTime = Date.now();
let isProcessingFrame = false; // ensure only one in-flight request at a time

// DOM elements
const startBtn = document.getElementById('start-btn');
const stopBtn = document.getElementById('stop-btn');
const resetBtn = document.getElementById('reset-btn');
const poseStatus = document.getElementById('pose-status');
const confidenceEl = document.getElementById('confidence');
const feedbackEl = document.getElementById('feedback');
const anglesPanel = document.getElementById('angles-panel');
const anglesContent = document.getElementById('angles-content');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    video = document.getElementById('video');
    canvas = document.getElementById('canvas');
    ctx = canvas.getContext('2d');
    
    // Generate unique session ID
    sessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    
    // Event listeners
    startBtn.addEventListener('click', startCamera);
    stopBtn.addEventListener('click', stopCamera);
    resetBtn.addEventListener('click', resetSession);
    
    // Tab switching
    setupTabs();
});

function setupTabs() {
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetTab = btn.getAttribute('data-tab');
            
            // Remove active class from all tabs and contents
            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(content => content.classList.remove('active'));
            
            // Add active class to clicked tab and corresponding content
            btn.classList.add('active');
            document.getElementById(`${targetTab}-tab`).classList.add('active');
        });
    });
}

async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 1280 },
                height: { ideal: 720 },
                facingMode: 'user'
            }
        });
        
        video.srcObject = stream;
        isStreaming = true;
        
        video.addEventListener('loadedmetadata', () => {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            startProcessing();
        });
        
        startBtn.disabled = true;
        stopBtn.disabled = false;
        poseStatus.textContent = 'Camera started';
        
    } catch (error) {
        console.error('Error accessing camera:', error);
        poseStatus.textContent = 'Error: Could not access camera';
        alert('Could not access camera. Please ensure you have granted camera permissions.');
    }
}

function stopCamera() {
    if (video.srcObject) {
        const tracks = video.srcObject.getTracks();
        tracks.forEach(track => track.stop());
        video.srcObject = null;
    }
    
    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
    }
    
    isStreaming = false;
    startBtn.disabled = false;
    stopBtn.disabled = true;
    poseStatus.textContent = 'Camera stopped';
    feedbackEl.textContent = '';
    confidenceEl.textContent = '';
    anglesPanel.style.display = 'none';
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function startProcessing() {
    if (!isStreaming) return;
    
    // Calculate FPS
    frameCount++;
    const currentTime = Date.now();
    if (currentTime - lastTime >= 1000) {
        fps = frameCount;
        frameCount = 0;
        lastTime = currentTime;
    }
    
    // Only start a new capture if no request is currently in-flight
    if (!isProcessingFrame) {
        captureAndProcess();
    }
    
    animationFrameId = requestAnimationFrame(startProcessing);
}

function captureAndProcess() {
    if (isProcessingFrame || !isStreaming) return;
    isProcessingFrame = true;

    // Draw video frame to canvas
    ctx.save();
    ctx.scale(-1, 1);
    ctx.drawImage(video, -canvas.width, 0, canvas.width, canvas.height);
    ctx.restore();
    
    // Convert canvas to base64
    const imageData = canvas.toDataURL('image/jpeg', 0.8);
    
    // Send to server
    fetch('/process_frame', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            image: imageData,
            session_id: sessionId,
            fps: fps
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            console.error('Server error:', data.error);
            poseStatus.textContent = 'Error: ' + data.error;
        } else {
            // Update UI with results
            updateUI(data);
        }
    })
    .catch(error => {
        console.error('Error processing frame:', error);
    })
    .finally(() => {
        isProcessingFrame = false;
    });
}

function updateUI(data) {
    // Update pose status
    poseStatus.textContent = data.pose_text;
    
    // Update confidence
    if (data.confidence > 0) {
        const confidencePercent = Math.round(data.confidence * 100);
        confidenceEl.textContent = `Confidence: ${confidencePercent}%`;
    } else {
        confidenceEl.textContent = '';
    }
    
    // Update feedback
    if (data.feedback_text) {
        feedbackEl.textContent = data.feedback_text;
        if (data.feedback_text.includes('Good')) {
            feedbackEl.classList.remove('needs-adjustment');
            feedbackEl.classList.add('good');
        } else {
            feedbackEl.classList.remove('good');
            feedbackEl.classList.add('needs-adjustment');
        }
    } else {
        feedbackEl.textContent = '';
    }
    
    // Update angles display
    if (data.has_pose && data.angles) {
        displayAngles(data.angles, data.visibility);
        anglesPanel.style.display = 'block';
    } else {
        anglesPanel.style.display = 'none';
    }
}

function displayAngles(angles, visibility) {
    anglesContent.innerHTML = '';
    
    // Get relevant angles based on current pose (simplified - show all)
    const angleKeys = Object.keys(angles);
    
    angleKeys.forEach(key => {
        const angleItem = document.createElement('div');
        angleItem.className = 'angle-item';
        
        const vis = visibility && visibility[key] ? visibility[key] : 1.0;
        const isLowVis = vis < 0.7;
        
        if (isLowVis) {
            angleItem.classList.add('low-visibility');
        }
        
        const angleValue = Math.round(angles[key]);
        angleItem.innerHTML = `
            <strong>${formatAngleName(key)}</strong><br>
            ${angleValue}° ${isLowVis ? '(low visibility)' : ''}
        `;
        
        anglesContent.appendChild(angleItem);
    });
}

function formatAngleName(key) {
    return key
        .replace(/_/g, ' ')
        .replace(/\b\w/g, l => l.toUpperCase());
}

async function resetSession() {
    try {
        await fetch('/reset_session', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                session_id: sessionId
            })
        });
        
        // Generate new session ID
        sessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
        
        poseStatus.textContent = 'Session reset';
        feedbackEl.textContent = '';
        confidenceEl.textContent = '';
        anglesPanel.style.display = 'none';
        
    } catch (error) {
        console.error('Error resetting session:', error);
    }
}

