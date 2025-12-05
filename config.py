"""
Shared configuration for Yoga Pose Detection.
Edit these values to tune behavior across both desktop and web apps.
"""

import os

# Core runtime parameters
CONFIDENCE_THRESH = 0.60           # Prediction confidence threshold (0.0-1.0)
WINDOW_SECONDS = 2              # Sliding window duration in seconds

# Resolve model path relative to the repository root (this config file's location)
_ROOT = os.path.dirname(__file__)
MODEL_PATH = os.path.join(_ROOT, 'models', 'yoga_pose_model_new.pkl')

# Temporal filtering
EMA_ALPHA = 0.2                    # EMA smoothing factor for probabilities (higher = smoother, slower to react)
MIN_STABLE_SECONDS = 0.75           # Minimum consecutive seconds before switching displayed label
POSE_HOLD_SECONDS = 3.0             # Minimum seconds a pose must be held before counting it

# Visibility gating
BODY_VIS_THRESH = 0.7              # Minimum visibility for key joints to consider body fully in frame
