"""
Shared configuration for Yoga Pose Detection.
Edit these values to tune behavior across both desktop and web apps.
"""

# Core runtime parameters
CONFIDENCE_THRESH = 0.60           # Prediction confidence threshold (0.0-1.0)
WINDOW_SECONDS = 1.5               # Sliding window duration in seconds
MODEL_PATH = 'yoga_pose_model0.pkl'  # Path to trained model file

# Temporal filtering
EMA_ALPHA = 0.2                    # EMA smoothing factor for probabilities (higher = smoother, slower to react)
MIN_STABLE_SECONDS = 0.6           # Minimum consecutive seconds before switching displayed label

# Visibility gating
BODY_VIS_THRESH = 0.7              # Minimum visibility for key joints to consider body fully in frame
