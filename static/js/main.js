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
let sessionActive = false;

// DOM elements
const startSessionBtn = document.getElementById('start-session-btn');
const stopSessionBtn = document.getElementById('stop-session-btn');
const poseStatus = document.getElementById('pose-status');
const confidenceEl = document.getElementById('confidence');
const feedbackEl = document.getElementById('feedback');
const anglesPanel = document.getElementById('angles-panel');
const anglesContent = document.getElementById('angles-content');
const sessionSummaryEl = document.getElementById('session-summary');
const sessionDurationEl = document.getElementById('session-duration');
const sessionPosesEl = document.getElementById('session-poses');
const sessionFeedbackEl = document.getElementById('session-feedback');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    video = document.getElementById('video');
    canvas = document.getElementById('canvas');
    ctx = canvas.getContext('2d');
    
    // Generate unique session ID
    sessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    
    // Event listeners
    startSessionBtn.addEventListener('click', startSession);
    stopSessionBtn.addEventListener('click', stopSession);
    
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

async function startSession() {
    if (sessionActive) return;
    
    try {
        // Start camera
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
        
        // Start session tracking on server
        await fetch('/start_session', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: sessionId })
        });
        
        sessionActive = true;
        startSessionBtn.disabled = true;
        stopSessionBtn.disabled = false;
        poseStatus.textContent = 'Session started';
        if (sessionSummaryEl) sessionSummaryEl.style.display = 'none';
        
    } catch (error) {
        console.error('Error starting session:', error);
        poseStatus.textContent = 'Error: Could not start session';
        alert('Could not access camera. Please ensure you have granted camera permissions.');
    }
}

async function stopSession() {
    if (!sessionActive) return;
    
    // Stop camera
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
    
    // End session and get summary
    try {
        const response = await fetch('/end_session', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: sessionId })
        });
        const summary = await response.json();
        renderSessionSummary(summary);
    } catch (error) {
        console.error('Error ending session:', error);
    }
    
    sessionActive = false;
    startSessionBtn.disabled = false;
    stopSessionBtn.disabled = true;
    poseStatus.textContent = 'Session stopped';
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
    // This ensures sequential processing without throttling (which was causing lag)
    if (!isProcessingFrame) {
        captureAndProcess();
    }
    
    animationFrameId = requestAnimationFrame(startProcessing);
}

function captureAndProcess() {
    if (isProcessingFrame || !isStreaming) return;
    isProcessingFrame = true;

    // Draw video frame to canvas (no flip - let CSS handle mirror display)
    // This ensures MediaPipe processes the same orientation as desktop app
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    // Convert canvas to base64 - use 0.7 quality for good balance
    const imageData = canvas.toDataURL('image/jpeg', 0.7);
    
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

function renderSessionSummary(summary) {
    if (!summary || !sessionSummaryEl) return;

    const duration = summary.duration_seconds != null ? summary.duration_seconds : null;
    if (duration != null && sessionDurationEl) {
        const mins = Math.floor(duration / 60);
        const secs = Math.round(duration % 60);
        sessionDurationEl.textContent = `Duration: ${mins}m ${secs}s`;
    }

    if (sessionPosesEl) {
        const poses = summary.pose_counts || {};
        const items = Object.entries(poses);
        if (items.length) {
            const ul = document.createElement('ul');
            items.forEach(([pose, count]) => {
                const li = document.createElement('li');
                li.textContent = `${pose}: ${count} reps`;
                ul.appendChild(li);
            });
            sessionPosesEl.innerHTML = '<strong>Poses:</strong>';
            sessionPosesEl.appendChild(ul);
        } else {
            sessionPosesEl.textContent = 'No poses recorded.';
        }
    }

    if (sessionFeedbackEl) {
        const fbSummary = summary.feedback_summary || {};
        const entries = Object.entries(fbSummary);
        if (entries.length) {
            const container = document.createElement('div');
            entries.forEach(([pose, messages]) => {
                const section = document.createElement('div');
                const title = document.createElement('div');
                title.innerHTML = `<strong>${pose} feedback:</strong>`;
                section.appendChild(title);
                const ul = document.createElement('ul');
                messages.forEach(m => {
                    const li = document.createElement('li');
                    li.textContent = `${m.message} (${m.count}x)`;
                    ul.appendChild(li);
                });
                section.appendChild(ul);
                container.appendChild(section);
            });
            sessionFeedbackEl.innerHTML = '';
            sessionFeedbackEl.appendChild(container);
        } else {
            sessionFeedbackEl.textContent = 'No corrective feedback recorded.';
        }
    }

    sessionSummaryEl.style.display = 'block';
}

