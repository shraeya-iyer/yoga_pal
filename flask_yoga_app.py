# Flask Web Application to show case our Real-time Yoga Pose Detection, serves web interface for yoga pose estimation and feedback

# only tracks in summary if we hold pose for 3 seconds

from flask import Flask, render_template, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque
import warnings
import os
import sys
import argparse
import time
import base64
from io import BytesIO
from PIL import Image

# Import functions and constants from azim's script (realtime_yoga_pose.py)
# suppress exit() calls that happen during import if model file doesn't exist
try:
    # temp replace sys.exit to prevent import from killing Flask app because realtime_yoga_pose.py calls exit(1) if model file doesn't exist
    original_exit = sys.exit
    
    def dummy_exit(code=0):
        """Dummy exit function that does nothing - prevents realtime_yoga_pose from killing Flask"""
        pass
    
    sys.exit = dummy_exit
    
    from realtime_yoga_pose import (
        calculate_angles_from_kp,
        extract_features_single_frame,
        reconstruct_angles,
        get_feedback,
        feedback_rules
    )
    print("Imported functions from realtime_yoga_pose.py")

    from realtime_yoga_pose import (CONFIDENCE_THRESH, WINDOW_SECONDS, MODEL_PATH)
    print("Imported constants from realtime_yoga_pose.py")

finally:
    # Restore original sys.exit
    sys.exit = original_exit

warnings.filterwarnings("ignore")

app = Flask(__name__)

# Load the trained model
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file '{MODEL_PATH}' not found - model should be in same dir as this script")
    model = None
else:
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")

# Initialize MediaPipe utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Store sliding windows per client (using session ID)
client_buffers = {}

# In-memory session tracking per session_id
session_state = {}  # {session_id: {active, start_time, end_time, pose_counts, feedback_counts, last_pose}}

# Reuse MediaPipe instances per session to improve performance
# Each session gets its own instance, avoiding timestamp issues while reducing overhead
pose_detectors = {}  # {session_id: Pose instance}

# Temporal filtering parameters (tunable)
EMA_ALPHA = 0.2  # EMA smoothing factor for probabilities
MIN_STABLE_SECONDS = 0.6  # Require this many seconds of consistent predictions before switching
BODY_VIS_THRESH = 0.7  # Minimum visibility for required joints to consider body fully in frame

def get_pose_detector(session_id):
    """Get or create a MediaPipe Pose instance for a session.
    Reusing instances per session improves performance while avoiding timestamp issues."""
    if session_id not in pose_detectors:
        pose_detectors[session_id] = mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    return pose_detectors[session_id]


@app.route('/')
def index():
    # Serve the main web interface
    return render_template('index.html')


@app.route('/process_frame', methods=['POST'])
def process_frame():
    #Process a single frame from the webcam
    if model is None:
        return jsonify({
            'error': 'Model not loaded. Please ensure model file exists.'
        }), 500
    
    try:
        data = request.json
        image_data = data.get('image')
        session_id = data.get('session_id', 'default')
        fps = data.get('fps', 30)
        
        # Decode base64 image
        image_data = image_data.split(',')[1]  # Remove data:image/jpeg;base64, prefix
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        # PIL Image is already RGB, convert directly to numpy array
        # No need for BGR conversion since browser sends RGB JPEG
        image_rgb = np.array(image)
        
        # Initialize buffer for this session if needed
        if session_id not in client_buffers:
            buffer_size = int(fps * WINDOW_SECONDS)
            client_buffers[session_id] = {
                'feature_buffer': deque(maxlen=buffer_size),
                'vis_buffer': deque(maxlen=buffer_size),
                'smoothed_probs': None,
                'current_label': None,
                'stable_candidate': None,
                'stable_frames': 0,
                'min_stable_frames': max(1, int(fps * MIN_STABLE_SECONDS))
            }
        
        buffer = client_buffers[session_id]
        
        # Initialize response
        response = {
            'pose_text': 'Buffering...',
            'feedback_text': '',
            'confidence': 0.0,
            'angles': {},
            'has_pose': False
        }
        
        # Process with MediaPipe (image_rgb is already RGB from PIL)
        # Note: Desktop app has flip commented out, so we don't flip here either
        # CSS handles mirror display for user experience
        # Reuse MediaPipe instance per session for better performance
        try:
            pose_detector = get_pose_detector(session_id)
            results = pose_detector.process(image_rgb)
        except Exception as mp_error:
            # Handle MediaPipe errors (like timestamp mismatches) gracefully
            print(f"MediaPipe processing error: {mp_error}")
            # Return a response indicating we're waiting for a valid frame
            response['pose_text'] = 'Waiting...'
            response['feedback_text'] = 'Processing frame...'
            return jsonify(response)

        if results and results.pose_world_landmarks:
            # Extract features from current frame
            raw_vec, angles, vis = extract_features_single_frame(
                results.pose_world_landmarks.landmark
            )
            
            # Add to sliding window buffers
            buffer['feature_buffer'].append(raw_vec)
            buffer['vis_buffer'].append(vis)
            
            # Prepare visibility averages
            avg_vis = {}
            if len(buffer['vis_buffer']) >= max(1, int(fps * 0.3)):
                for key in vis:
                    avg_vis[key] = float(np.mean([v[key] for v in buffer['vis_buffer']]))

            def body_fully_visible(vis_dict, required_keys=None, thresh=BODY_VIS_THRESH):
                if not vis_dict:
                    return False
                if required_keys is None:
                    required_keys = [
                        'left_shoulder_angle', 'right_shoulder_angle',
                        'left_hip_angle', 'right_hip_angle',
                        'left_knee_angle', 'right_knee_angle',
                    ]
                for k in required_keys:
                    if vis_dict.get(k, 0.0) < thresh:
                        return False
                return True

            vis_ok = body_fully_visible(vis) or (avg_vis and body_fully_visible(avg_vis))

            # Once buffer is full, make predictions
            if len(buffer['feature_buffer']) == buffer['feature_buffer'].maxlen and vis_ok:
                # Average features over sliding window for stability
                avg_vec = np.mean(buffer['feature_buffer'], axis=0)
                
                # Predict pose
                probabilities = model.predict_proba(avg_vec.reshape(1, -1))[0]
                # Apply EMA smoothing to probabilities (per session)
                if buffer['smoothed_probs'] is None:
                    buffer['smoothed_probs'] = probabilities.copy()
                else:
                    buffer['smoothed_probs'] = EMA_ALPHA * probabilities + (1 - EMA_ALPHA) * buffer['smoothed_probs']

                confidence = float(np.max(buffer['smoothed_probs']))
                pred_idx = int(np.argmax(buffer['smoothed_probs']))
                pred_label = model.classes_[pred_idx]
                
                # Only show prediction if confidence is high enough
                if confidence > CONFIDENCE_THRESH:
                    # Debounce/hysteresis
                    if buffer['current_label'] is None:
                        buffer['current_label'] = pred_label
                        buffer['stable_candidate'] = pred_label
                        buffer['stable_frames'] = 1
                    else:
                        if pred_label == buffer['current_label']:
                            buffer['stable_candidate'] = pred_label
                            buffer['stable_frames'] = min(buffer['stable_frames'] + 1, buffer['min_stable_frames'])
                        else:
                            if buffer['stable_candidate'] == pred_label:
                                buffer['stable_frames'] += 1
                            else:
                                buffer['stable_candidate'] = pred_label
                                buffer['stable_frames'] = 1
                            if buffer['stable_frames'] >= buffer['min_stable_frames']:
                                buffer['current_label'] = pred_label
                                buffer['stable_frames'] = 0

                    response['pose_text'] = f"{buffer['current_label']}"
                    response['confidence'] = confidence
                    response['has_pose'] = True
                    
                    # avg_vis already computed above
                    
                    avg_angles_dict = reconstruct_angles(avg_vec)
                    
                    # Get pose-specific feedback
                    feedback_text = get_feedback(buffer['current_label'], avg_angles_dict, avg_vis)
                    response['feedback_text'] = feedback_text
                    response['angles'] = {k: float(v) for k, v in avg_angles_dict.items()}
                    response['visibility'] = avg_vis

                    # Session tracking (optimized - minimal overhead)
                    # Only count a "rep" when pose changes, not on every frame
                    if session_id in session_state:
                        sess = session_state[session_id]
                        if sess.get("active"):
                            last_pose = sess.get("last_pose")
                            # Only increment rep count if pose changed (new rep)
                            if last_pose != pred_label:
                                pose_counts = sess["pose_counts"]
                                pose_counts[pred_label] = pose_counts.get(pred_label, 0) + 1
                                sess["last_pose"] = pred_label  # Update last pose
                            
                            # Track only corrective feedback (can accumulate across frames)
                            if feedback_text and feedback_text != "Good job!":
                                feedback_counts = sess["feedback_counts"]
                                if pred_label not in feedback_counts:
                                    feedback_counts[pred_label] = {}
                                feedback_counts[pred_label][feedback_text] = feedback_counts[pred_label].get(feedback_text, 0) + 1
                else:
                    response['pose_text'] = 'Waiting...'
                    response['feedback_text'] = 'Please hold a pose'
            elif len(buffer['feature_buffer']) == buffer['feature_buffer'].maxlen and not vis_ok:
                response['pose_text'] = 'Please put whole body in frame'
                response['feedback_text'] = 'Ensure shoulders, hips, and knees are visible'
                response['has_pose'] = False
            else:
                # Still filling buffer
                buffer_progress = len(buffer['feature_buffer'])
                buffer_size = buffer['feature_buffer'].maxlen
                response['pose_text'] = f'Buffering... ({buffer_progress}/{buffer_size})'
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'error': f'Error processing frame: {str(e)}'
        }), 500


@app.route('/start_session', methods=['POST'])
def start_session():
    """Start a yoga session for a given session_id."""
    data = request.json or {}
    sess_id = data.get('session_id', 'default')
    session_state[sess_id] = {
        'active': True,
        'start_time': time.time(),
        'end_time': None,
        'pose_counts': {},
        'feedback_counts': {},
        'last_pose': None  # Track last detected pose to count reps correctly
    }
    # Create MediaPipe instance for this session (will be reused)
    if sess_id not in pose_detectors:
        pose_detectors[sess_id] = mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    return jsonify({'status': 'ok'})


@app.route('/end_session', methods=['POST'])
def end_session():
    """End a yoga session and return a summary."""
    data = request.json or {}
    sess_id = data.get('session_id', 'default')
    sess = session_state.get(sess_id)
    if not sess:
        return jsonify({'error': 'Session not found'}), 400

    sess['active'] = False
    if sess.get('end_time') is None:
        sess['end_time'] = time.time()

    duration = None
    if sess.get('start_time') is not None and sess.get('end_time') is not None:
        duration = sess['end_time'] - sess['start_time']

    pose_counts = sess.get('pose_counts', {})
    feedback_counts = sess.get('feedback_counts', {})

    # Build feedback summary: top messages per pose
    feedback_summary = {}
    for pose_name, fb_dict in feedback_counts.items():
        items = sorted(fb_dict.items(), key=lambda x: x[1], reverse=True)
        feedback_summary[pose_name] = [
            {'message': msg, 'count': count} for msg, count in items
        ]

    # Clean up MediaPipe instance for this session
    if sess_id in pose_detectors:
        try:
            pose_detectors[sess_id].close()
        except Exception:
            pass
        del pose_detectors[sess_id]

    summary = {
        'duration_seconds': duration,
        'pose_counts': pose_counts,
        'feedback_summary': feedback_summary,
    }
    return jsonify(summary)


@app.route('/reset_session', methods=['POST'])
def reset_session():
    # Reset the sliding window buffer and session state for a session
    data = request.json
    session_id = data.get('session_id', 'default')
    
    if session_id in client_buffers:
        del client_buffers[session_id]
    if session_id in session_state:
        del session_state[session_id]
    if session_id in pose_detectors:
        try:
            pose_detectors[session_id].close()
        except Exception:
            pass
        del pose_detectors[session_id]
    
    return jsonify({'status': 'reset'})


if __name__ == '__main__':
    # Parse command line arguments for port
    parser = argparse.ArgumentParser(description='Yoga Pose Detection Web App')
    parser.add_argument('--port', type=int, default=5001,
                        help='Port to run the Flask server on (default: 5001)') # since on macos port 5000 is usually used by airplay receivers
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Host to bind to (default: 0.0.0.0)')
    args = parser.parse_args()
    
    # Also check for PORT environment variable
    port = int(os.environ.get('PORT', args.port))
    host = os.environ.get('HOST', args.host)
    
    print("Yoga Pose Detection Web App")
    print("Starting Flask server...")
    print(f"Access the app at: http://localhost:{port}")
    print("Press Ctrl+C to stop the server")
    
    app.run(debug=True, host=host, port=port)

