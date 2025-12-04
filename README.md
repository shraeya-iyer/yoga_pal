# Real-time Yoga Pose Detection

A real-time yoga pose estimation and feedback system using webcam, MediaPipe, and machine learning.

## This is the final project for Northwestern course CS_396: Machine Learning and Sensing.

## By Shraeya Iyer, Azim Usmanov, Brendan Tan-Fahed, Irina Djuraskovirina

## Features

- **Real-time pose detection** from webcam feed
- **Pose classification** for 4 yoga poses: Tree, Lunge, Cobra, Downward Dog
- **Live feedback** on pose form with specific corrections
- **Joint angle visualization** for debugging
- **Sliding window smoothing** for stable predictions
- **Temporal filtering** (EMA + debounce) to reduce jitter between pose changes
- **Mirror mode** for intuitive practice

## Setup

### 1. Install Dependencies

Ensure you have Python installed, then run:

```bash
pip install -r requirements.txt
```

### 2. Ensure Model File Exists

The script requires `yoga_pose_model.pkl` to be in the same directory. This file should be generated from the notebook's training section.

If it is not included, run the model training cells in the .ipynb notebook (up to cells 28-29) to generate it.

### 3. Run the Application

You can run the application in two ways:

#### Option A: Web Interface (Recommended)

Run the Flask web app:

```bash
python flask_yoga_app.py
or
python3 flask_yoga_app.py
```

Then open your browser and navigate to:
```
http://localhost:5000
```

#### Option B: Desktop Application

Run the OpenCV-based desktop app:

```bash
python realtime_yoga_pose.py 
or
python3 realtime_yoga_pose.py 
```

## Usage

Once the script is running:

1. **Position yourself** so your full body is visible in the webcam
2. **Hold a yoga pose** for 1 second (the buffering period)
3. **View feedback** displayed on screen:
   - Green text = Good form
   - Red text = Needs adjustment
4. **Check joint angles** displayed on the left side (grayed out = low visibility)

### Keyboard Controls

- **`q`** - Quit the application
- **`s`** - Save a screenshot of the current frame

## Supported Poses

### 1. Tree Pose
- Bent knee: 30-85°
- Standing leg: 165-185° (straight)

### 2. Lunge
- Front knee: 75-110° (bent at ~90°)
- Back leg: 160-185° (straight)

### 3. Cobra Pose
- Back arch: 90-145°
- Arms: 150-185° (mostly straight)

### 4. Downward Dog
- Legs: 160-185° (straight)
- Hips: 40-110° (bent)
- Arms: 140-185° (straight)
- Chest: 135-195°

## Troubleshooting

### Camera not opening
- Make sure no other application is using the webcam
- Try changing camera index in code: `cv2.VideoCapture(0)` → `cv2.VideoCapture(1)`

### Low accuracy
- Ensure full body is visible in frame
- Good lighting helps MediaPipe detect landmarks
- Hold poses still for at least 1 second
- Adjust `CONFIDENCE_THRESH` in the script (default: 0.65)

### Slow performance
- Reduce MediaPipe model complexity: change `model_complexity=1` to `model_complexity=0` (line 30)
- Reduce window size: `WINDOW_SECONDS = 0.5` (line 17)
 - Increase EMA smoothing or stability window to trade responsiveness for stability

## Configuration

You can adjust these parameters at the top of `realtime_yoga_pose.py`:

```python
CONFIDENCE_THRESH = 0.65  # Prediction confidence threshold (0.0-1.0)
WINDOW_SECONDS = 1        # Sliding window duration in seconds
MODEL_PATH = 'yoga_pose_model.pkl'  # Path to trained model
EMA_ALPHA = 0.2           # Temporal EMA smoothing factor for probabilities (higher = smoother)
MIN_STABLE_SECONDS = 0.6  # Minimum consecutive seconds to commit a new pose label
```

## How It Works

1. **Capture**: Grabs frames from webcam at ~30 FPS
2. **Detection**: MediaPipe extracts 33 body keypoints
3. **Feature Extraction**: Calculates 8 joint angles from keypoints
4. **Smoothing**: Averages angles over a 1-second sliding window
5. **Classification**: Random Forest model predicts pose
6. **Temporal Filtering**: EMA smooths probabilities and a debounce requires stable predictions before switching labels
7. **Feedback**: Rule-based system checks angle ranges and provides tips
8. **Display**: Overlays skeleton, pose name, confidence, and feedback

### Tuning Temporal Filtering

- Make pose changes faster: decrease `MIN_STABLE_SECONDS` or decrease `EMA_ALPHA`.
- Make pose changes steadier: increase `MIN_STABLE_SECONDS` or increase `EMA_ALPHA`.
- Typical values: `EMA_ALPHA` in `0.15–0.35`, `MIN_STABLE_SECONDS` in `0.4–1.0`.

## Web Interface

The web interface provides a modern, browser-based experience for yoga pose detection.

### Features

- **Modern UI**: Clean, responsive design with gradient backgrounds
- **Real-time Processing**: Live webcam feed with instant pose detection
- **Visual Feedback**: Color-coded feedback (green for good form, red for adjustments)
- **Joint Angles Display**: Real-time visualization of joint angles
- **Mobile-Friendly**: Responsive design that works on various screen sizes
- **Mirror Mode**: Automatically flips the video for intuitive practice

### Usage

1. Start the Flask server: `python flask_yoga_app.py`
2. Open your browser to `http://localhost:5000`
3. Click "Start Camera" and allow camera permissions
4. Position yourself so your full body is visible
5. Hold a pose for 1 second to receive feedback
6. View real-time pose classification and feedback

### Web Interface Controls

- **Start Camera**: Begin webcam feed and pose detection
- **Stop Camera**: Stop the webcam and processing
- **Reset**: Clear the sliding window buffer and start fresh

### Architecture

The web app uses:
- **Backend**: Flask server processing frames server-side
- **Frontend**: HTML5 video API for webcam access
- **Communication**: RESTful API with JSON responses
- **Processing**: Same MediaPipe and ML model as desktop version

---

**Note**: This script uses a mirror view (horizontally flipped) for more intuitive practice. Your right side appears on the right side of the screen.
