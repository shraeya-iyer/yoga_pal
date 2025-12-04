"""
Real-time Yoga Pose Estimation and Feedback System
Uses webcam to detect yoga poses and provide live feedback
"""

import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque
import warnings
import os

warnings.filterwarnings("ignore")

# Configuration
CONFIDENCE_THRESH = 0.60
# CONFIDENCE_THRESH = 0.40 # Way too low
WINDOW_SECONDS = 1.5  # Sliding window duration
# MODEL_PATH = 'yoga_pose_model.pkl'
MODEL_PATH = 'yoga_pose_model0.pkl'
EMA_ALPHA = 0.2  # Exponential moving average smoothing factor for probabilities (0-1)
MIN_STABLE_SECONDS = 0.6  # Require this many seconds of consistent predictions before switching pose


# Load the trained model
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file '{MODEL_PATH}' not found!")
    print("Please ensure yoga_pose_model.pkl is in the same directory as this script.")
    exit(1)

model = joblib.load(MODEL_PATH)

# Initialize MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False, 
    model_complexity=1,  # 0=Lite, 1=Full, 2=Heavy (use 1 for balance)
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Feedback rules for each pose
feedback_rules = {
    "Lunge": {
        "front_knee": (75, 110, "Bend front knee to 90°"),
        "back_knee": (150, 185, "Straighten back leg"),
    },
    "Tree": {
        "bent_knee": (30, 85, "Bend knee more"),
        "standing_knee": (165, 185, "Straighten standing leg"), # should add arms for tree pose
    },
    "DownwardDog": {
        "legs": (160, 185, "Straighten legs"),
        "back": (40, 110, "Bend hip more"),
        "arms": (140, 185, "Straighten arms"),
        "chest": (135, 195, "Tuck chest in more")
    },
    "Cobra": {
        "back": (90, 145, "Arch back more"),
        "arms": (150, 185, "Straighten arms")
    }
}


def calculate_angles_from_kp(kp1, kp2, kp3, offset=0):
    """Calculate angle between three keypoints"""
    kp1 = np.array(kp1)
    kp2 = np.array(kp2)
    kp3 = np.array(kp3)
    
    vec1 = kp1 - kp2
    vec2 = kp3 - kp2
    
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    cosine_ang = dot / (norm1 * norm2)
    cosine_ang = np.clip(cosine_ang, -1.0, 1.0)
    angle = np.arccos(cosine_ang)
    d_angle = np.degrees(angle)
    
    return d_angle + offset


def extract_features_single_frame(landmarks):
    """Extract joint angles and visibility from pose landmarks"""
    def get_coords(idx):
        kp = landmarks[idx]
        return [kp.x, kp.y, kp.z]

    # MediaPipe joint indices
    L_SHOULDER, R_SHOULDER = 11, 12
    L_ELBOW, R_ELBOW = 13, 14
    L_WRIST, R_WRIST = 15, 16
    L_HIP, R_HIP = 23, 24
    L_KNEE, R_KNEE = 25, 26
    L_ANKLE, R_ANKLE = 27, 28

    # Calculate all joint angles
    angles = {}
    angles['left_elbow_angle'] = calculate_angles_from_kp(
        get_coords(L_SHOULDER), get_coords(L_ELBOW), get_coords(L_WRIST))
    angles['right_elbow_angle'] = calculate_angles_from_kp(
        get_coords(R_SHOULDER), get_coords(R_ELBOW), get_coords(R_WRIST))
    angles['left_knee_angle'] = calculate_angles_from_kp(
        get_coords(L_HIP), get_coords(L_KNEE), get_coords(L_ANKLE))
    angles['right_knee_angle'] = calculate_angles_from_kp(
        get_coords(R_HIP), get_coords(R_KNEE), get_coords(R_ANKLE))
    angles['left_hip_angle'] = calculate_angles_from_kp(
        get_coords(L_SHOULDER), get_coords(L_HIP), get_coords(L_KNEE))
    angles['right_hip_angle'] = calculate_angles_from_kp(
        get_coords(R_SHOULDER), get_coords(R_HIP), get_coords(R_KNEE))
    angles['left_shoulder_angle'] = calculate_angles_from_kp(
        get_coords(L_HIP), get_coords(L_SHOULDER), get_coords(L_ELBOW))
    angles['right_shoulder_angle'] = calculate_angles_from_kp(
        get_coords(R_HIP), get_coords(R_SHOULDER), get_coords(R_ELBOW))

    # Get visibility scores for each joint
    vis = {}
    vis['left_elbow_angle'] = landmarks[L_ELBOW].visibility
    vis['right_elbow_angle'] = landmarks[R_ELBOW].visibility
    vis['left_knee_angle'] = landmarks[L_KNEE].visibility
    vis['right_knee_angle'] = landmarks[R_KNEE].visibility
    vis['left_hip_angle'] = landmarks[L_HIP].visibility
    vis['right_hip_angle'] = landmarks[R_HIP].visibility
    vis['left_shoulder_angle'] = landmarks[L_SHOULDER].visibility
    vis['right_shoulder_angle'] = landmarks[R_SHOULDER].visibility

    # Create feature vector (same order as training)
    angle_features = [
        angles['left_elbow_angle'], angles['right_elbow_angle'],
        angles['left_knee_angle'], angles['right_knee_angle'],
        angles['left_hip_angle'], angles['right_hip_angle'],
        angles['left_shoulder_angle'], angles['right_shoulder_angle']
    ]
    
    return np.array(angle_features), angles, vis


def reconstruct_angles(feature_vector):
    """Convert feature vector back to named angles dictionary"""
    return {
        'left_elbow_angle': feature_vector[0],
        'right_elbow_angle': feature_vector[1],
        'left_knee_angle': feature_vector[2],
        'right_knee_angle': feature_vector[3],
        'left_hip_angle': feature_vector[4],
        'right_hip_angle': feature_vector[5],
        'left_shoulder_angle': feature_vector[6],
        'right_shoulder_angle': feature_vector[7]
    }


def get_feedback(pose_name, angles_dict, vis_dict):
    """Generate feedback based on pose-specific rules"""
    if pose_name not in feedback_rules:
        return ""
    
    rules = feedback_rules[pose_name]
    l_knee = angles_dict['left_knee_angle']
    r_knee = angles_dict['right_knee_angle']
    mapping = {}

    # Map generic joint names to left/right specific angles
    if pose_name == 'Tree':
        if l_knee < r_knee:
            mapping = {'bent_knee': 'left_knee_angle', 'standing_knee': 'right_knee_angle'}
        else:
            mapping = {'bent_knee': 'right_knee_angle', 'standing_knee': 'left_knee_angle'}

    elif pose_name == 'Lunge':
        if l_knee < r_knee:
            mapping = {'front_knee': 'left_knee_angle', 'back_knee': 'right_knee_angle'}
        else:
            mapping = {'front_knee': 'right_knee_angle', 'back_knee': 'left_knee_angle'}

    elif pose_name == 'DownwardDog':
        mapping['legs'] = ['left_knee_angle', 'right_knee_angle']
        mapping['back'] = ['left_hip_angle', 'right_hip_angle']
        mapping['arms'] = ['left_elbow_angle', 'right_elbow_angle']
        mapping['chest'] = ['left_shoulder_angle', 'right_shoulder_angle']

    elif pose_name == 'Cobra':
        mapping['back'] = ['left_hip_angle', 'right_hip_angle']
        mapping['arms'] = ['left_elbow_angle', 'right_elbow_angle']

    visual_thresh = 0.85 # prev 0.7

    # Check each rule
    for rule_part, (min_v, max_v, msg) in rules.items():
        names = mapping.get(rule_part)
        if not names:
            continue
        if not isinstance(names, list):
            names = [names]

        for name in names:
            # Skip if joint not visible enough
            if vis_dict and vis_dict.get(name, 0) < visual_thresh:
                continue
            
            val = angles_dict.get(name, 0)
            if val < min_v or val > max_v:
                return msg

    return "Good job!"


def run_realtime_pose_detection():
    """Main function to run real-time pose detection from webcam"""
    # Open webcam (0 is usually the default camera)

    # IMPORTANT !!!!!!!!!!!!!!!!!!!!





    # MIGHT NEED TO CHANGE THIS LATER TO WORK WITH IPHONE
    cap = cv2.VideoCapture(0)
    



























    if not cap.isOpened():
        print("Error: Could not open webcam!")
        print("Please check that your camera is connected and not in use by another application.")
        return

    # Get camera properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        fps = 30  # Default if unable to get FPS
    
    # Calculate buffer size for sliding window
    buffer_size = int(fps * WINDOW_SECONDS)
    feature_buffer = deque(maxlen=buffer_size)
    vis_buffer = deque(maxlen=buffer_size)

    print("=" * 60)
    print("Real-time Yoga Pose Detection Started!")
    print("=" * 60)
    print(f"Camera FPS: {fps}")
    print(f"Sliding window: {WINDOW_SECONDS} second(s) ({buffer_size} frames)")
    print(f"Confidence threshold: {CONFIDENCE_THRESH * 100}%")
    print("\nSupported poses: Tree, Lunge, Cobra, DownwardDog")
    print("\nControls:")
    print("  - Press 'q' to quit")
    print("  - Press 's' to save current frame as screenshot")
    print("=" * 60)

    frame_count = 0
    screenshot_count = 0

    try:
        # Temporal filtering state
        smoothed_probs = None  # EMA of class probabilities
        current_label = None
        stable_candidate = None
        stable_frames = 0
        min_stable_frames = max(1, int(fps * MIN_STABLE_SECONDS))

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to grab frame from webcam")
                break

            frame_count += 1
            
            # Flip frame horizontally for mirror effect (more intuitive)
            # frame = cv2.flip(frame, 1)
            
            # Get frame dimensions
            height, width = frame.shape[:2]

            # Convert to RGB for MediaPipe
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            # Initialize display variables
            pose_text = "Buffering..."
            feedback_text = ""
            debug_angles = {}
            avg_vis = {}
            confidence = 0.0
            pred_label = None

            # Process pose landmarks if detected
            if results.pose_world_landmarks:
                # Draw skeleton on frame
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, 
                        results.pose_landmarks, 
                        mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                    )
                
                # Extract features from current frame
                raw_vec, angles, vis = extract_features_single_frame(
                    results.pose_world_landmarks.landmark
                )

                # Add to sliding window buffers
                feature_buffer.append(raw_vec)
                vis_buffer.append(vis)

                # Once buffer is full, make predictions
                if len(feature_buffer) == buffer_size:
                    # Average features over sliding window for stability
                    avg_vec = np.mean(feature_buffer, axis=0)

                    # Predict pose
                    probabilities = model.predict_proba(avg_vec.reshape(1, -1))[0]
                    # Apply EMA smoothing to probabilities
                    if smoothed_probs is None:
                        smoothed_probs = probabilities.copy()
                    else:
                        smoothed_probs = EMA_ALPHA * probabilities + (1 - EMA_ALPHA) * smoothed_probs

                    confidence = float(np.max(smoothed_probs))
                    pred_idx = int(np.argmax(smoothed_probs))
                    pred_label = model.classes_[pred_idx]

                    # Only show prediction if confidence is high enough
                    if confidence > CONFIDENCE_THRESH:
                        # Debounce/hysteresis: require consecutive frames before switching
                        if current_label is None:
                            current_label = pred_label
                            stable_candidate = pred_label
                            stable_frames = 1
                        else:
                            if pred_label == current_label:
                                # already stable on this label
                                stable_candidate = pred_label
                                stable_frames = min(stable_frames + 1, min_stable_frames)
                            else:
                                # considering switch: count consecutive frames for candidate
                                if stable_candidate == pred_label:
                                    stable_frames += 1
                                else:
                                    stable_candidate = pred_label
                                    stable_frames = 1

                                # switch only if we've seen enough consecutive frames
                                if stable_frames >= min_stable_frames:
                                    current_label = pred_label
                                    stable_frames = 0  # reset counter after switch

                        pose_text = f"{current_label} ({int(confidence * 100)}%)"

                        # Calculate average visibility
                        for key in vis:
                            avg_vis[key] = np.mean([v[key] for v in vis_buffer])
                        
                        avg_angles_dict = reconstruct_angles(avg_vec)

                        # Get pose-specific feedback
                        feedback_text = get_feedback(current_label, avg_angles_dict, avg_vis)

                        # Store angles for debug display
                        debug_angles = avg_angles_dict
                        pred_label = current_label  # for debug keys below
                    else:
                        pose_text = "Waiting..."
                        feedback_text = "Please hold a pose"
                else:
                    # Still filling buffer
                    pose_text = f"Buffering... ({len(feature_buffer)}/{buffer_size})"

            # Draw UI overlay
            # Top black bar for pose and feedback
            cv2.rectangle(frame, (0, 0), (width, 100), (0, 0, 0), -1)

            # Pose text
            status_color = (255, 255, 255)
            if pose_text == "Waiting...":
                status_color = (192, 192, 192)
            
            cv2.putText(frame, f"Pose: {pose_text}", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

            # Feedback text (green if good, red if needs adjustment)
            color = (0, 255, 0) if "Good" in feedback_text else (0, 0, 255)
            cv2.putText(frame, f"{feedback_text}", (20, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

            # Display relevant joint angles for debugging
            if debug_angles and pose_text != "Waiting..." and pred_label:
                y_offset = 150
                relevant_keys = []

                # Show different angles based on detected pose
                if pred_label == 'Cobra':
                    relevant_keys = ['left_hip_angle', 'right_hip_angle', 
                                   'left_elbow_angle', 'right_elbow_angle']
                elif pred_label == 'Tree':
                    relevant_keys = ['left_knee_angle', 'right_knee_angle']
                elif pred_label == 'Lunge':
                    relevant_keys = ['left_knee_angle', 'right_knee_angle']
                elif pred_label == 'DownwardDog':
                    relevant_keys = ['left_hip_angle', 'right_hip_angle', 
                                   'left_knee_angle', 'right_knee_angle',
                                   'left_elbow_angle', 'right_elbow_angle', 
                                   'left_shoulder_angle', 'right_shoulder_angle']

                # Draw each relevant angle
                for key in relevant_keys:
                    val = debug_angles.get(key, 0)
                    vis_score = avg_vis.get(key, 0)
                    
                    # Gray text if visibility is low
                    if vis_score < 0.7:
                        text_color = (128, 128, 128)
                    else:
                        text_color = (255, 255, 0)
                    
                    cv2.putText(frame, f"{key}: {int(val)}° (v:{vis_score:.2f})",
                               (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.6, text_color, 2)
                    y_offset += 30

            # Add instructions at bottom
            cv2.putText(frame, "Press 'q' to quit | 's' to save screenshot", 
                       (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (255, 255, 255), 1)

            # Display the frame
            cv2.imshow('Yoga Pose Detection', frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('s'):
                screenshot_count += 1
                filename = f'screenshot_{screenshot_count}.jpg'
                cv2.imwrite(filename, frame)
                print(f"Screenshot saved: {filename}")

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError occurred: {e}")
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        pose.close()
        print("Webcam released and windows closed.")
        print(f"Total frames processed: {frame_count}")


if __name__ == "__main__":
    run_realtime_pose_detection()
