import cv2  # type: ignore
import numpy as np  # type: ignore
import tensorflow as tf  # type: ignore
import time
from collections import deque

# Suppress TensorFlow warnings for cleaner output
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load the trained model
model = tf.keras.models.load_model('eye_open_close_model.keras')

# Constants
IMG_SIZE = (64, 64)
CONFIDENCE_THRESHOLD = 0.5
SMOOTHING_WINDOW = 5  # frames for smoothing predictions

# Haar cascades for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Statistics tracking
class EyeTracker:
    def __init__(self):
        self.frame_count = 0
        self.faces_detected = 0
        self.eyes_detected = 0
        self.closed_eye_count = 0
        self.open_eye_count = 0
        self.start_time = time.time()
        self.fps_history = deque(maxlen=30)
        self.eye_state_history = deque(maxlen=SMOOTHING_WINDOW)
    
    def update_fps(self):
        current_time = time.time()
        if hasattr(self, 'last_time'):
            fps = 1.0 / (current_time - self.last_time)
            self.fps_history.append(fps)
        self.last_time = current_time
    
    def get_avg_fps(self):
        return np.mean(self.fps_history) if self.fps_history else 0
    
    def smooth_prediction(self, prediction):
        """Smooth predictions to reduce flickering"""
        self.eye_state_history.append(prediction)
        return np.mean(self.eye_state_history)

# Initialize tracker
tracker = EyeTracker()

# Color definitions
COLORS = {
    'open': (0, 255, 0),      # 🟢 Green
    'closed': (0, 0, 255),    # 🔴 Red
    'mixed': (0, 165, 255),   # 🟠 Orange
    'no_eyes': (255, 0, 0),   # 🔵 Blue
    'text': (255, 255, 255),  # White
    'bg': (0, 0, 0)           # Black
}

def preprocess_eye(eye_region):
    """Optimized eye preprocessing"""
    try:
        eye_resized = cv2.resize(eye_region, IMG_SIZE)
        eye_normalized = eye_resized.astype('float32') / 255.0
        eye_input = np.expand_dims(eye_normalized, axis=(0, -1))
        return eye_input
    except Exception:
        return None

def predict_eye_state(eye_input, confidence_threshold=CONFIDENCE_THRESHOLD):
    """Predict eye state with confidence"""
    try:
        prediction = model.predict(eye_input, verbose=0)[0][0]
        confidence = abs(prediction - 0.5) * 2  # Convert to 0-1 confidence scale
        is_open = prediction > confidence_threshold
        return is_open, prediction, confidence
    except Exception:
        return None, 0.0, 0.0

def draw_info_panel(frame, tracker, face_count, eye_count):
    """Draw statistics panel"""
    height, width = frame.shape[:2]
    panel_height = 120
    panel_width = 300
    
    # Create semi-transparent overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (panel_width, panel_height), COLORS['bg'], -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    
    # Statistics text
    stats = [
        f"FPS: {tracker.get_avg_fps():.1f}",
        f"Faces: {face_count}",
        f"Eyes: {eye_count}",
        f"Total Frames: {tracker.frame_count}",
        f"Open Eyes: {tracker.open_eye_count}",
        f"Closed Eyes: {tracker.closed_eye_count}"
    ]
    
    for i, stat in enumerate(stats):
        y_pos = 25 + i * 15
        cv2.putText(frame, stat, (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS['text'], 1)

def get_face_status(eye_states):
    """Determine overall face status"""
    if not eye_states:
        return "No Eyes Detected", COLORS['no_eyes']
    
    open_count = eye_states.count(True)
    closed_count = eye_states.count(False)
    
    if closed_count >= 2:
        return "Sleepy - Both Eyes Closed", COLORS['closed']
    elif closed_count == 1 and open_count >= 1:
        return "Drowsy - One Eye Closed", COLORS['mixed']
    elif open_count >= 1:
        return "Awake - Eyes Open", COLORS['open']
    else:
        return "Unknown State", COLORS['no_eyes']

# Start webcam with optimized settings
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

print("🎥 Enhanced Real-time Eye & Head State Tracking")
print("Features:")
print("🟢 Green: Eyes Open | 🔴 Red: Eyes Closed | 🟠 Orange: Mixed State | 🔵 Blue: No Eyes")
print("Press 'q' to quit, 'r' to reset statistics")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Update frame statistics
    tracker.frame_count += 1
    tracker.update_fps()
    
    # Convert to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces with optimized parameters
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.2, 
        minNeighbors=5, 
        minSize=(50, 50),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    face_count = len(faces)
    total_eyes = 0
    
    for (x, y, w, h) in faces:
        tracker.faces_detected += 1
        
        # Define regions of interest
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        # Detect eyes within face region
        eyes = eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(20, 20)
        )
        
        eye_states = []
        eye_confidences = []
        
        # Process each detected eye
        for (ex, ey, ew, eh) in eyes:
            total_eyes += 1
            tracker.eyes_detected += 1
            
            # Extract and preprocess eye region
            eye_region = roi_gray[ey:ey+eh, ex:ex+ew]
            eye_input = preprocess_eye(eye_region)
            
            if eye_input is not None:
                # Predict eye state
                is_open, prediction, confidence = predict_eye_state(eye_input)
                
                # Smooth prediction to reduce flickering
                smoothed_prediction = tracker.smooth_prediction(prediction)
                is_open_smoothed = smoothed_prediction > CONFIDENCE_THRESHOLD
                
                eye_states.append(is_open_smoothed)
                eye_confidences.append(confidence)
                
                # Update statistics
                if is_open_smoothed:
                    tracker.open_eye_count += 1
                else:
                    tracker.closed_eye_count += 1
                
                # Draw eye bounding box
                color = COLORS['open'] if is_open_smoothed else COLORS['closed']
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), color, 2)
                
                # Draw eye state label with confidence
                label = f"{'Open' if is_open_smoothed else 'Closed'} ({confidence:.2f})"
                cv2.putText(roi_color, label, (ex, ey-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Determine face status and color
        face_status, face_color = get_face_status(eye_states)
        
        # Draw face bounding box
        cv2.rectangle(frame, (x, y), (x+w, y+h), face_color, 3)
        
        # Draw face status label
        cv2.putText(frame, face_status, (x, y-15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, face_color, 2)
        
        # Draw eye count for this face
        eye_info = f"Eyes: {len(eyes)}"
        cv2.putText(frame, eye_info, (x, y+h+20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, face_color, 1)
    
    # Draw information panel
    draw_info_panel(frame, tracker, face_count, total_eyes)
    
    # Display frame
    cv2.imshow('Enhanced Eye & Head Tracking', frame)
    
    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        # Reset statistics
        tracker = EyeTracker()
        print("📊 Statistics reset!")

# Cleanup
cap.release()
cv2.destroyAllWindows()

# Print final statistics
print(f"\n📊 Final Statistics:")
print(f"Total Runtime: {time.time() - tracker.start_time:.1f} seconds")
print(f"Total Frames: {tracker.frame_count}")
print(f"Average FPS: {tracker.get_avg_fps():.1f}")
print(f"Faces Detected: {tracker.faces_detected}")
print(f"Eyes Detected: {tracker.eyes_detected}")
print(f"Open Eyes: {tracker.open_eye_count}")
print(f"Closed Eyes: {tracker.closed_eye_count}")
print("👋 Thank you for using Enhanced Eye Tracking!")



















#iteration 2


# import cv2  # type: ignore
# import numpy as np  # type: ignore
# import tensorflow as tf  # type: ignore
# import time
# from collections import deque
# from dataclasses import dataclass
# from typing import List, Tuple, Optional

# # Suppress TensorFlow warnings for cleaner output
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# # Load the trained model
# model = tf.keras.models.load_model('eye_open_close_model.keras')

# # Constants
# IMG_SIZE = (64, 64)
# CONFIDENCE_THRESHOLD = 0.5
# SMOOTHING_WINDOW = 3
# BLINK_THRESHOLD = 0.3  # Lower threshold for blink detection
# BLINK_CONSECUTIVE_FRAMES = 2  # Minimum frames for valid blink
# PERCLOS_WINDOW = 30  # seconds for PERCLOS calculation
# MIN_BLINK_SEPARATION = 5  # Minimum frames between blinks

# @dataclass
# class BlinkEvent:
#     """Represents a single blink event"""
#     start_frame: int
#     end_frame: int
#     duration: float
#     timestamp: float

# @dataclass
# class EyeState:
#     """Represents the state of a single eye"""
#     is_open: bool
#     confidence: float
#     prediction_value: float
#     bounding_box: Tuple[int, int, int, int]

# class AdvancedEyeTracker:
#     def __init__(self):
#         # Basic statistics
#         self.frame_count = 0
#         self.faces_detected = 0
#         self.eyes_detected = 0
#         self.start_time = time.time()
        
#         # Eye state tracking
#         self.left_eye_open_count = 0
#         self.left_eye_closed_count = 0
#         self.right_eye_open_count = 0
#         self.right_eye_closed_count = 0
        
#         # FPS tracking
#         self.fps_history = deque(maxlen=30)
#         self.last_time = time.time()
        
#         # Blink detection
#         self.blink_history = []
#         self.left_eye_state_history = deque(maxlen=10)
#         self.right_eye_state_history = deque(maxlen=10)
#         self.both_eyes_state_history = deque(maxlen=10)
#         self.last_blink_frame = 0
#         self.current_blink_start = None
#         self.total_blinks = 0
#         self.blinks_per_minute = 0
        
#         # PERCLOS calculation
#         self.perclos_history = deque(maxlen=int(PERCLOS_WINDOW * 30))  # 30 FPS assumption
#         self.perclos_value = 0.0
#         self.drowsiness_level = "Alert"
        
#         # Prediction smoothing
#         self.left_eye_predictions = deque(maxlen=SMOOTHING_WINDOW)
#         self.right_eye_predictions = deque(maxlen=SMOOTHING_WINDOW)
    
#     def update_fps(self):
#         """Update FPS calculation"""
#         current_time = time.time()
#         fps = 1.0 / (current_time - self.last_time)
#         self.fps_history.append(fps)
#         self.last_time = current_time
    
#     def get_avg_fps(self):
#         """Get average FPS"""
#         return np.mean(self.fps_history) if self.fps_history else 0
    
#     def smooth_prediction(self, prediction: float, eye_side: str) -> float:
#         """Smooth predictions to reduce noise"""
#         if eye_side == 'left':
#             self.left_eye_predictions.append(prediction)
#             return np.mean(self.left_eye_predictions)
#         else:
#             self.right_eye_predictions.append(prediction)
#             return np.mean(self.right_eye_predictions)
    
#     def detect_blink(self, left_eye_open: bool, right_eye_open: bool) -> bool:
#         """
#         Detect simultaneous blinks (both eyes must close and open together)
#         Returns True if a complete blink is detected
#         """
#         both_eyes_closed = not left_eye_open and not right_eye_open
#         self.both_eyes_state_history.append(both_eyes_closed)
        
#         # Need at least 5 frames of history
#         if len(self.both_eyes_state_history) < 5:
#             return False
        
#         # Check for blink pattern: open -> closed -> open
#         states = list(self.both_eyes_state_history)
        
#         # Prevent counting blinks too frequently
#         if self.frame_count - self.last_blink_frame < MIN_BLINK_SEPARATION:
#             return False
        
#         # Look for blink pattern in recent history
#         for i in range(len(states) - 4):
#             # Pattern: open, closed, closed, open (minimum)
#             if (not states[i] and      # was open
#                 states[i+1] and        # became closed
#                 states[i+2] and        # stayed closed
#                 not states[i+3]):      # became open again
                
#                 # Mark blink detected
#                 self.last_blink_frame = self.frame_count
#                 self.total_blinks += 1
                
#                 # Calculate blinks per minute
#                 elapsed_minutes = (time.time() - self.start_time) / 60
#                 if elapsed_minutes > 0:
#                     self.blinks_per_minute = self.total_blinks / elapsed_minutes
                
#                 return True
        
#         return False
    
#     def update_perclos(self, both_eyes_closed: bool):
#         """Update PERCLOS (Percentage of Eyelid Closure) calculation"""
#         self.perclos_history.append(both_eyes_closed)
        
#         if len(self.perclos_history) >= 30:  # At least 1 second of data
#             closed_frames = sum(self.perclos_history)
#             total_frames = len(self.perclos_history)
#             self.perclos_value = (closed_frames / total_frames) * 100
            
#             # Determine drowsiness level based on PERCLOS
#             if self.perclos_value >= 80:
#                 self.drowsiness_level = "Severely Drowsy"
#             elif self.perclos_value >= 50:
#                 self.drowsiness_level = "Moderately Drowsy"
#             elif self.perclos_value >= 20:
#                 self.drowsiness_level = "Mildly Drowsy"
#             else:
#                 self.drowsiness_level = "Alert"
    
#     def get_drowsiness_color(self):
#         """Get color based on drowsiness level"""
#         colors = {
#             "Alert": (0, 255, 0),           # Green
#             "Mildly Drowsy": (0, 255, 255), # Yellow
#             "Moderately Drowsy": (0, 165, 255), # Orange
#             "Severely Drowsy": (0, 0, 255)  # Red
#         }
#         return colors.get(self.drowsiness_level, (255, 255, 255))

# # Initialize tracker
# tracker = AdvancedEyeTracker()

# # Haar cascades for face and eye detection
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# # Color definitions
# COLORS = {
#     'open': (0, 255, 0),      # Green
#     'closed': (0, 0, 255),    # Red
#     'mixed': (0, 165, 255),   # Orange
#     'no_eyes': (255, 0, 0),   # Blue
#     'text': (255, 255, 255),  # White
#     'bg': (0, 0, 0),          # Black
#     'blink': (255, 0, 255)    # Magenta for blink indication
# }

# def preprocess_eye(eye_region):
#     """Optimized eye preprocessing"""
#     try:
#         eye_resized = cv2.resize(eye_region, IMG_SIZE)
#         eye_normalized = eye_resized.astype('float32') / 255.0
#         eye_input = np.expand_dims(eye_normalized, axis=(0, -1))
#         return eye_input
#     except Exception:
#         return None

# def predict_eye_state(eye_input, confidence_threshold=CONFIDENCE_THRESHOLD):
#     """Predict eye state with confidence"""
#     try:
#         prediction = model.predict(eye_input, verbose=0)[0][0]
#         confidence = abs(prediction - 0.5) * 2
#         is_open = prediction > confidence_threshold
#         return is_open, prediction, confidence
#     except Exception:
#         return None, 0.0, 0.0

# def classify_eyes(eyes, roi_gray):
#     """Classify eyes as left/right based on position"""
#     if len(eyes) != 2:
#         return None, None
    
#     # Sort eyes by x-coordinate (left eye has smaller x)
#     eyes_sorted = sorted(eyes, key=lambda eye: eye[0])
#     left_eye = eyes_sorted[0]
#     right_eye = eyes_sorted[1]
    
#     return left_eye, right_eye

# def draw_advanced_info_panel(frame, tracker):
#     """Draw comprehensive statistics panel"""
#     height, width = frame.shape[:2]
#     panel_height = 200
#     panel_width = 350
    
#     # Create semi-transparent overlay
#     overlay = frame.copy()
#     cv2.rectangle(overlay, (10, 10), (panel_width, panel_height), COLORS['bg'], -1)
#     cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    
#     # Statistics text
#     stats = [
#         f"FPS: {tracker.get_avg_fps():.1f}",
#         f"Frame: {tracker.frame_count}",
#         f"Runtime: {time.time() - tracker.start_time:.1f}s",
#         "",
#         f"👁️ Eye Statistics:",
#         f"Left Eye - Open: {tracker.left_eye_open_count}",
#         f"Left Eye - Closed: {tracker.left_eye_closed_count}",
#         f"Right Eye - Open: {tracker.right_eye_open_count}",
#         f"Right Eye - Closed: {tracker.right_eye_closed_count}",
#         "",
#         f"👀 Blink Analysis:",
#         f"Total Blinks: {tracker.total_blinks}",
#         f"Blinks/Min: {tracker.blinks_per_minute:.1f}",
#         "",
#         f"😴 PERCLOS Analysis:",
#         f"PERCLOS: {tracker.perclos_value:.1f}%",
#         f"Status: {tracker.drowsiness_level}"
#     ]
    
#     for i, stat in enumerate(stats):
#         if stat:  # Skip empty strings
#             y_pos = 25 + i * 12
#             color = tracker.get_drowsiness_color() if "Status:" in stat else COLORS['text']
#             cv2.putText(frame, stat, (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

# def draw_blink_indicator(frame, blink_detected):
#     """Draw blink indicator"""
#     if blink_detected:
#         height, width = frame.shape[:2]
#         cv2.putText(frame, "BLINK DETECTED!", (width - 200, 30), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['blink'], 2)
#         cv2.circle(frame, (width - 30, 30), 15, COLORS['blink'], -1)

# # Start webcam with optimized settings
# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# cap.set(cv2.CAP_PROP_FPS, 30)
# cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# print("👁️ Advanced Eye Tracking System with Blink Detection & PERCLOS")
# print("Features:")
# print("🟢 Green: Eyes Open | 🔴 Red: Eyes Closed")
# print("👀 Simultaneous Blink Detection")
# print("😴 PERCLOS Drowsiness Analysis")
# print("📊 Individual Eye Statistics")
# print("Press 'q' to quit, 'r' to reset statistics")

# blink_detected = False
# blink_timer = 0

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
    
#     # Update frame statistics
#     tracker.frame_count += 1
#     tracker.update_fps()
    
#     # Reset blink indicator
#     if blink_timer > 0:
#         blink_timer -= 1
#         if blink_timer == 0:
#             blink_detected = False
    
#     # Convert to grayscale for detection
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
#     # Detect faces
#     faces = face_cascade.detectMultiScale(
#         gray, 
#         scaleFactor=1.2, 
#         minNeighbors=5, 
#         minSize=(50, 50),
#         flags=cv2.CASCADE_SCALE_IMAGE
#     )
    
#     for (x, y, w, h) in faces:
#         tracker.faces_detected += 1
        
#         # Define regions of interest
#         roi_gray = gray[y:y+h, x:x+w]
#         roi_color = frame[y:y+h, x:x+w]
        
#         # Detect eyes within face region
#         eyes = eye_cascade.detectMultiScale(
#             roi_gray,
#             scaleFactor=1.1,
#             minNeighbors=3,
#             minSize=(20, 20)
#         )
        
#         # Process exactly 2 eyes for proper blink detection
#         if len(eyes) == 2:
#             left_eye, right_eye = classify_eyes(eyes, roi_gray)
            
#             # Process left eye
#             ex, ey, ew, eh = left_eye
#             eye_region = roi_gray[ey:ey+eh, ex:ex+ew]
#             eye_input = preprocess_eye(eye_region)
            
#             left_eye_open = True
#             left_confidence = 0.0
            
#             if eye_input is not None:
#                 is_open, prediction, confidence = predict_eye_state(eye_input, BLINK_THRESHOLD)
#                 smoothed_prediction = tracker.smooth_prediction(prediction, 'left')
#                 left_eye_open = smoothed_prediction > BLINK_THRESHOLD
#                 left_confidence = confidence
                
#                 # Update statistics
#                 if left_eye_open:
#                     tracker.left_eye_open_count += 1
#                 else:
#                     tracker.left_eye_closed_count += 1
                
#                 # Draw left eye
#                 color = COLORS['open'] if left_eye_open else COLORS['closed']
#                 cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), color, 2)
#                 cv2.putText(roi_color, f"L: {'Open' if left_eye_open else 'Closed'}", 
#                            (ex, ey-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
#             # Process right eye
#             ex, ey, ew, eh = right_eye
#             eye_region = roi_gray[ey:ey+eh, ex:ex+ew]
#             eye_input = preprocess_eye(eye_region)
            
#             right_eye_open = True
#             right_confidence = 0.0
            
#             if eye_input is not None:
#                 is_open, prediction, confidence = predict_eye_state(eye_input, BLINK_THRESHOLD)
#                 smoothed_prediction = tracker.smooth_prediction(prediction, 'right')
#                 right_eye_open = smoothed_prediction > BLINK_THRESHOLD
#                 right_confidence = confidence
                
#                 # Update statistics
#                 if right_eye_open:
#                     tracker.right_eye_open_count += 1
#                 else:
#                     tracker.right_eye_closed_count += 1
                
#                 # Draw right eye
#                 color = COLORS['open'] if right_eye_open else COLORS['closed']
#                 cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), color, 2)
#                 cv2.putText(roi_color, f"R: {'Open' if right_eye_open else 'Closed'}", 
#                            (ex, ey-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
#             # Detect blinks (both eyes must close simultaneously)
#             if tracker.detect_blink(left_eye_open, right_eye_open):
#                 blink_detected = True
#                 blink_timer = 15  # Show blink indicator for 15 frames
            
#             # Update PERCLOS
#             both_eyes_closed = not left_eye_open and not right_eye_open
#             tracker.update_perclos(both_eyes_closed)
            
#             # Determine face status
#             if both_eyes_closed:
#                 face_status = "Both Eyes Closed"
#                 face_color = COLORS['closed']
#             elif not left_eye_open or not right_eye_open:
#                 face_status = "One Eye Closed"
#                 face_color = COLORS['mixed']
#             else:
#                 face_status = f"Alert - {tracker.drowsiness_level}"
#                 face_color = tracker.get_drowsiness_color()
        
#         else:
#             # Handle cases with != 2 eyes
#             face_status = f"Eyes Detected: {len(eyes)}"
#             face_color = COLORS['no_eyes']
        
#         # Draw face bounding box
#         cv2.rectangle(frame, (x, y), (x+w, y+h), face_color, 3)
#         cv2.putText(frame, face_status, (x, y-15), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, face_color, 2)
    
#     # Draw information panels
#     draw_advanced_info_panel(frame, tracker)
#     draw_blink_indicator(frame, blink_detected)
    
#     # Display frame
#     cv2.imshow('Advanced Eye Tracking with Blink Detection & PERCLOS', frame)
    
#     # Handle key presses
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('q'):
#         break
#     elif key == ord('r'):
#         # Reset statistics
#         tracker = AdvancedEyeTracker()
#         print("📊 All statistics reset!")

# # Cleanup
# cap.release()
# cv2.destroyAllWindows()

# # Print final comprehensive report
# print(f"\n📊 COMPREHENSIVE FINAL REPORT")
# print(f"{'='*50}")
# print(f"⏱️  Runtime: {time.time() - tracker.start_time:.1f} seconds")
# print(f"🎬 Total Frames: {tracker.frame_count}")
# print(f"📈 Average FPS: {tracker.get_avg_fps():.1f}")
# print(f"\n👁️  EYE STATISTICS:")
# print(f"   Left Eye  - Open: {tracker.left_eye_open_count:4d} | Closed: {tracker.left_eye_closed_count:4d}")
# print(f"   Right Eye - Open: {tracker.right_eye_open_count:4d} | Closed: {tracker.right_eye_closed_count:4d}")
# print(f"\n👀 BLINK ANALYSIS:")
# print(f"   Total Blinks: {tracker.total_blinks}")
# print(f"   Blinks per Minute: {tracker.blinks_per_minute:.1f}")
# print(f"   Normal Range: 12-20 blinks/min")
# print(f"\n😴 PERCLOS DROWSINESS ANALYSIS:")
# print(f"   Final PERCLOS: {tracker.perclos_value:.1f}%")
# print(f"   Drowsiness Level: {tracker.drowsiness_level}")
# print(f"   PERCLOS Scale: <20% Alert | 20-50% Mild | 50-80% Moderate | >80% Severe")
# print(f"\n👋 Session Complete - Thank you for using Advanced Eye Tracking!")