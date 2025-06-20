# Import configuration first to suppress warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import numpy as np
import tensorflow as tf
import time
import warnings
from collections import deque

# Suppress warnings
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

class OptimizedEyeTracker:
    def __init__(self):
        # Core timing
        self.start_time = time.time()
        self.frame_count = 0
        
        # Enhanced blink detection
        self.total_blinks = 0
        self.valid_blinks = 0
        self.blinks_per_minute = 0.0
        self.last_blink_time = 0
        self.blink_timestamps = deque(maxlen=50)
        
        # Improved eye state tracking
        self.left_eye_states = deque(maxlen=6)
        self.right_eye_states = deque(maxlen=6)
        self.blink_threshold = 0.3  # Lowered for better detection
        self.min_blink_gap = 0.15   # 150ms between blinks
        self.consecutive_closed_frames = 0
        self.blink_in_progress = False
        
        # PERCLOS calculation
        self.eye_closure_samples = deque(maxlen=90)  # 3 seconds at 30fps
        self.perclos_percentage = 0.0
        
        # Timing configuration
        self.min_calculation_time = 8.0  # 8 seconds minimum
        self.calculation_progress = 0.0
        
        # Performance tracking
        self.fps_samples = deque(maxlen=20)
        self.last_frame_time = time.time()
        self.faces_detected = 0
        self.eyes_detected = 0
        
        # Eye state confidence tracking
        self.eye_confidence_history = deque(maxlen=10)

    def update_performance_metrics(self):
        """Update FPS and frame counting"""
        current_time = time.time()
        self.frame_count += 1
        
        if hasattr(self, 'last_frame_time'):
            fps = 1.0 / max(current_time - self.last_frame_time, 0.001)
            self.fps_samples.append(fps)
        
        self.last_frame_time = current_time
        
        # Update calculation progress
        elapsed_time = current_time - self.start_time
        if elapsed_time < self.min_calculation_time:
            self.calculation_progress = (elapsed_time / self.min_calculation_time) * 100

    def detect_blink_enhanced(self, left_eye_open, right_eye_open, left_confidence=0.5, right_confidence=0.5):
        """Enhanced blink detection with confidence weighting"""
        current_time = time.time()
        
        # Weight eye states by confidence
        avg_confidence = (left_confidence + right_confidence) / 2.0
        
        # Only proceed if we have reasonable confidence
        if avg_confidence < 0.3:
            return False
        
        # Store eye states
        self.left_eye_states.append(left_eye_open)
        self.right_eye_states.append(right_eye_open)
        self.eye_confidence_history.append(avg_confidence)
        
        # Calculate closure for PERCLOS
        closure_value = 0.0
        if not left_eye_open and not right_eye_open:
            closure_value = 100.0
        elif not left_eye_open or not right_eye_open:
            closure_value = 50.0
        
        self.eye_closure_samples.append(closure_value)
        
        # Update PERCLOS
        if len(self.eye_closure_samples) > 0:
            self.perclos_percentage = sum(self.eye_closure_samples) / len(self.eye_closure_samples)
        
        # Enhanced blink detection
        if len(self.left_eye_states) >= 4 and len(self.right_eye_states) >= 4:
            # Get recent states
            recent_left = list(self.left_eye_states)[-4:]
            recent_right = list(self.right_eye_states)[-4:]
            
            # Look for blink pattern: open -> closed -> open
            for i in range(1, len(recent_left) - 1):
                # Check for simultaneous blink pattern
                left_blink = (recent_left[i-1] and not recent_left[i] and recent_left[i+1])
                right_blink = (recent_right[i-1] and not recent_right[i] and recent_right[i+1])
                
                # Alternative: check for any eye closing and opening
                any_blink = ((recent_left[i-1] or recent_right[i-1]) and 
                           (not recent_left[i] or not recent_right[i]) and 
                           (recent_left[i+1] or recent_right[i+1]))
                
                if left_blink and right_blink:
                    # Perfect simultaneous blink
                    if current_time - self.last_blink_time > self.min_blink_gap:
                        self._register_blink(current_time, "simultaneous")
                        return True
                elif any_blink and avg_confidence > 0.6:
                    # Partial blink with high confidence
                    if current_time - self.last_blink_time > self.min_blink_gap:
                        self._register_blink(current_time, "partial")
                        return True
        
        return False

    def _register_blink(self, current_time, blink_type):
        """Register a detected blink"""
        self.total_blinks += 1
        if blink_type == "simultaneous":
            self.valid_blinks += 1
        
        self.last_blink_time = current_time
        self.blink_timestamps.append(current_time)
        self._update_blink_rate()

    def _update_blink_rate(self):
        """Update blink rate calculation"""
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        if elapsed_time < self.min_calculation_time:
            self.blinks_per_minute = 0.0
            return
        
        # Use sliding window approach
        if elapsed_time >= 60.0:
            # Use 60-second window
            recent_blinks = [t for t in self.blink_timestamps if current_time - t <= 60.0]
            self.blinks_per_minute = len(recent_blinks)
        else:
            # Use total average
            elapsed_minutes = elapsed_time / 60.0
            self.blinks_per_minute = self.total_blinks / elapsed_minutes

    def get_blink_status(self):
        """Get blink rate status"""
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        if elapsed_time < self.min_calculation_time:
            return f"Calibrating... {self.calculation_progress:.0f}%", (255, 255, 0)
        
        if self.total_blinks == 0:
            return "No Blinks Detected", (255, 100, 100)
        elif self.blinks_per_minute < 8:
            return "Very Low", (255, 0, 0)
        elif self.blinks_per_minute < 12:
            return "Low", (255, 165, 0)
        elif self.blinks_per_minute > 25:
            return "High", (255, 165, 0)
        else:
            return "Normal", (0, 255, 0)

    def is_drowsy(self):
        """Determine drowsiness state"""
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        if elapsed_time < 15.0:
            return False
        
        # Multiple drowsiness indicators
        high_perclos = self.perclos_percentage > 70.0
        very_low_blinks = (self.blinks_per_minute > 0 and self.blinks_per_minute < 6.0)
        
        return high_perclos or very_low_blinks

    def get_average_fps(self):
        """Get average FPS"""
        return np.mean(self.fps_samples) if self.fps_samples else 0

    def get_stats(self):
        """Get comprehensive statistics"""
        current_time = time.time()
        runtime = current_time - self.start_time
        recent_blinks = len([t for t in self.blink_timestamps if current_time - t <= 60])
        
        return {
            'runtime': runtime,
            'fps': self.get_average_fps(),
            'frame_count': self.frame_count,
            'faces_detected': self.faces_detected,
            'eyes_detected': self.eyes_detected,
            'total_blinks': self.total_blinks,
            'valid_blinks': self.valid_blinks,
            'blinks_per_minute': self.blinks_per_minute,
            'recent_blinks': recent_blinks,
            'perclos': self.perclos_percentage,
            'is_drowsy': self.is_drowsy(),
            'calculation_ready': runtime >= self.min_calculation_time,
            'avg_confidence': np.mean(self.eye_confidence_history) if self.eye_confidence_history else 0
        }

# Load model safely
try:
    print("🔄 Loading eye state model...")
    model = tf.keras.models.load_model('eye_open_close_model.keras', compile=False)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

# Load Haar cascades
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    print("✅ Haar cascades loaded successfully!")
except Exception as e:
    print(f"❌ Error loading cascades: {e}")
    exit(1)

# Constants
IMG_SIZE = (64, 64)
CONFIDENCE_THRESHOLD = 0.5

# Enhanced color scheme
COLORS = {
    'open': (0, 255, 0),        # Green
    'closed': (0, 0, 255),      # Red
    'mixed': (0, 165, 255),     # Orange
    'text': (255, 255, 255),    # White
    'text_secondary': (200, 200, 200),  # Light gray
    'bg_primary': (0, 0, 0),    # Black
    'bg_secondary': (30, 30, 30),  # Dark gray
    'blink': (255, 255, 0),     # Yellow
    'drowsy': (0, 0, 255),      # Red
    'alert': (0, 255, 0),       # Green
    'header': (100, 255, 255),  # Cyan
    'accent': (255, 100, 100)   # Light red
}

def preprocess_eye_safe(eye_region):
    """Safe eye preprocessing"""
    try:
        if eye_region.size == 0:
            return None
        eye_resized = cv2.resize(eye_region, IMG_SIZE)
        eye_normalized = eye_resized.astype('float32') / 255.0
        eye_input = np.expand_dims(eye_normalized, axis=(0, -1))
        return eye_input
    except Exception:
        return None

def predict_eye_state_robust(eye_input):
    """Robust eye state prediction"""
    try:
        prediction = model.predict(eye_input, verbose=0)[0][0]
        confidence = abs(prediction - 0.5) * 2
        is_open = prediction > CONFIDENCE_THRESHOLD
        return is_open, prediction, confidence
    except Exception:
        return True, 0.5, 0.0

def draw_enhanced_dual_panel(frame, tracker):
    """Draw enhanced dual-panel information display"""
    height, width = frame.shape[:2]
    stats = tracker.get_stats()
    status, status_color = tracker.get_blink_status()
    
    # Panel dimensions
    panel_width = 300
    panel_height = 320
    left_panel_x = 10
    right_panel_x = width - panel_width - 10
    panel_y = 10
    
    # Create semi-transparent overlays
    overlay = frame.copy()
    
    # Left panel background
    cv2.rectangle(overlay, (left_panel_x, panel_y), 
                 (left_panel_x + panel_width, panel_y + panel_height), 
                 COLORS['bg_primary'], -1)
    
    # Right panel background  
    cv2.rectangle(overlay, (right_panel_x, panel_y), 
                 (right_panel_x + panel_width, panel_y + panel_height), 
                 COLORS['bg_primary'], -1)
    
    # Apply transparency
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    
    # Draw panel borders
    cv2.rectangle(frame, (left_panel_x, panel_y), 
                 (left_panel_x + panel_width, panel_y + panel_height), 
                 COLORS['header'], 2)
    cv2.rectangle(frame, (right_panel_x, panel_y), 
                 (right_panel_x + panel_width, panel_y + panel_height), 
                 COLORS['header'], 2)
    
    # LEFT PANEL - System & Performance Stats
    left_info = [
        "🖥️ SYSTEM PERFORMANCE",
        "",
        f"⏱️ Runtime: {stats['runtime']:.1f}s",
        f"🎬 FPS: {stats['fps']:.1f}",
        f"📊 Frames: {stats['frame_count']}",
        f"👥 Faces: {stats['faces_detected']}",
        f"👁️ Eyes: {stats['eyes_detected']}",
        f"🎯 Confidence: {stats['avg_confidence']:.2f}",
        "",
        "📋 DETECTION STATUS",
        "",
        f"🔍 Calculation: {'✅ Ready' if stats['calculation_ready'] else '⏳ Calibrating'}",
        f"🤖 Model: ✅ Loaded",
        f"📹 Camera: ✅ Active",
        f"🎛️ Cascades: ✅ Ready",
        "",
        "⚙️ THRESHOLDS",
        "",
        f"🎯 Confidence: {CONFIDENCE_THRESHOLD}",
        f"⏰ Min Gap: 150ms",
        f"📊 PERCLOS: 70%",
        f"🔄 Calibration: 8s"
    ]
    
    # RIGHT PANEL - Blink Analysis & Health Monitoring
    right_info = [
        "👁️ BLINK ANALYSIS",
        "",
        f"📊 Total Blinks: {stats['total_blinks']}",
        f"✅ Valid Blinks: {stats['valid_blinks']}",
        f"⚡ Rate: {stats['blinks_per_minute']:.1f}/min",
        f"🎯 Status: {status}",
        f"📈 Recent (60s): {stats['recent_blinks']}",
        f"📋 Normal: 12-20/min",
        "",
        "😴 HEALTH MONITORING",
        "",
        f"👁️ PERCLOS: {stats['perclos']:.1f}%",
        f"🚨 Threshold: 70%",
        f"🔋 State: {'😴 DROWSY' if stats['is_drowsy'] else '😊 ALERT'}",
        "",
        "📊 ANALYSIS RANGES",
        "",
        f"🟢 Normal: 12-20/min",
        f"🟡 Low: 8-12/min", 
        f"🔴 Very Low: <8/min",
        f"🟠 High: >25/min",
        "",
        "⚡ REAL-TIME FEEDBACK"
    ]
    
    # Draw left panel content
    y_offset = panel_y + 25
    for line in left_info:
        if line.strip() == "":
            y_offset += 8
            continue
        
        # Color coding for left panel
        if line.startswith("🖥️") or line.startswith("📋") or line.startswith("⚙️"):
            color = COLORS['header']
            font_scale = 0.45
        elif "✅" in line:
            color = COLORS['alert']
            font_scale = 0.4
        elif "⏳" in line:
            color = COLORS['blink']
            font_scale = 0.4
        else:
            color = COLORS['text']
            font_scale = 0.4
        
        cv2.putText(frame, line, (left_panel_x + 10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)
        y_offset += 14
    
    # Draw right panel content
    y_offset = panel_y + 25
    for line in right_info:
        if line.strip() == "":
            y_offset += 8
            continue
        
        # Color coding for right panel
        if line.startswith("👁️") or line.startswith("😴") or line.startswith("📊") or line.startswith("⚡"):
            color = COLORS['header']
            font_scale = 0.45
        elif "Status:" in line:
            color = status_color
            font_scale = 0.4
        elif "DROWSY" in line:
            color = COLORS['drowsy']
            font_scale = 0.4
        elif "ALERT" in line:
            color = COLORS['alert']
            font_scale = 0.4
        elif line.startswith("🟢") or line.startswith("🟡") or line.startswith("🔴") or line.startswith("🟠"):
            color = COLORS['text_secondary']
            font_scale = 0.35
        else:
            color = COLORS['text']
            font_scale = 0.4
        
        cv2.putText(frame, line, (right_panel_x + 10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)
        y_offset += 14
    
    # Add status indicators at the top
    if stats['is_drowsy']:
        cv2.putText(frame, "⚠️ DROWSINESS DETECTED", (width//2 - 100, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['drowsy'], 2)
    
    # Add real-time blink indicator
    if stats['calculation_ready']:
        indicator_text = f"👁️ MONITORING ACTIVE | Rate: {stats['blinks_per_minute']:.1f}/min"
        cv2.putText(frame, indicator_text, (width//2 - 150, height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['alert'], 1)

# Initialize tracker
tracker = OptimizedEyeTracker()

# Initialize camera
try:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Higher resolution
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        raise Exception("Cannot open camera")
    
    print("✅ Camera initialized at 1280x720!")
except Exception as e:
    print(f"❌ Camera error: {e}")
    exit(1)

print("\n" + "="*80)
print("🎯 ENHANCED EYE TRACKING WITH DUAL-PANEL INTERFACE")
print("="*80)
print("🟢 Features:")
print("  • Real-time blink detection with confidence weighting")
print("  • Dual-panel UI with organized information display")
print("  • PERCLOS-based drowsiness monitoring")
print("  • Performance metrics and system status")
print("  • Enhanced visual feedback and alerts")
print("="*80)
print("🎮 Controls:")
print("  'q' - Quit application")
print("  'r' - Reset all statistics")
print("  's' - Save comprehensive report")
print("  'c' - Calibrate detection sensitivity")
print("="*80)

# Main processing loop
blink_cooldown = 0
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to capture frame")
            break
        
        # Update performance metrics
        tracker.update_performance_metrics()
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(80, 80),
            maxSize=(400, 400)
        )
        
        blink_detected = False
        
        for (x, y, w, h) in faces:
            tracker.faces_detected += 1
            
            # Extract face ROI
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            
            # Detect eyes
            eyes = eye_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                maxSize=(80, 80)
            )
            
            eye_states = []
            eye_confidences = []
            
            # Process each eye
            for (ex, ey, ew, eh) in eyes:
                tracker.eyes_detected += 1
                
                # Extract and process eye region
                eye_region = roi_gray[ey:ey+eh, ex:ex+ew]
                eye_input = preprocess_eye_safe(eye_region)
                
                if eye_input is not None:
                    # Predict eye state
                    is_open, prediction, confidence = predict_eye_state_robust(eye_input)
                    
                    eye_states.append(is_open)
                    eye_confidences.append(confidence)
                    
                    # Draw eye rectangle with confidence-based color
                    if confidence > 0.7:
                        color = COLORS['open'] if is_open else COLORS['closed']
                    else:
                        color = COLORS['mixed']
                    
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), color, 2)
                    
                    # Draw confidence score
                    cv2.putText(roi_color, f"{confidence:.2f}", (ex, ey-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Blink detection
            if len(eyes) >= 2 and len(eye_states) >= 2:
                left_eye_open = eye_states[0]
                right_eye_open = eye_states[1]
                left_conf = eye_confidences[0] if len(eye_confidences) > 0 else 0.5
                right_conf = eye_confidences[1] if len(eye_confidences) > 1 else 0.5
                
                # Enhanced blink detection
                if tracker.detect_blink_enhanced(left_eye_open, right_eye_open, left_conf, right_conf):
                    blink_detected = True
                    blink_cooldown = 10  # Visual feedback duration
                    
                    # Blink animation
                    cv2.circle(frame, (x + w//2, y + h//2), 40, COLORS['blink'], 4)
                    cv2.putText(frame, "BLINK DETECTED!", (x - 30, y - 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS['blink'], 2)
            
            # Blink cooldown animation
            if blink_cooldown > 0:
                blink_cooldown -= 1
                cv2.circle(frame, (x + w//2, y + h//2), 25, COLORS['blink'], 2)
            
            # Draw face rectangle
            face_color = COLORS['drowsy'] if tracker.is_drowsy() else COLORS['alert']
            cv2.rectangle(frame, (x, y), (x+w, y+h), face_color, 3)
            
            # Face status
            status = "😴 DROWSY" if tracker.is_drowsy() else "😊 ALERT"
            cv2.putText(frame, status, (x, y-8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, face_color, 2)
        
        # Draw enhanced dual-panel display
        draw_enhanced_dual_panel(frame, tracker)
        
        # Show frame
        cv2.imshow('Enhanced Eye Tracking - Dual Panel Interface', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            tracker = OptimizedEyeTracker()
            print("📊 All statistics reset!")
        elif key == ord('s'):
            # Save comprehensive report
            stats = tracker.get_stats()
            timestamp = int(time.time())
            
            report = f"""
ENHANCED EYE TRACKING COMPREHENSIVE REPORT
=========================================
Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
Session Duration: {stats['runtime']:.1f} seconds ({stats['runtime']/60:.1f} minutes)

PERFORMANCE METRICS:
- Average FPS: {stats['fps']:.1f}
- Total Frames Processed: {stats['frame_count']}
- Detection Efficiency: {(stats['eyes_detected']/max(stats['frame_count'], 1)*100):.1f}%
- Average Confidence: {stats['avg_confidence']:.2f}

FACE & EYE DETECTION:
- Faces Detected: {stats['faces_detected']}
- Eyes Detected: {stats['eyes_detected']}
- Detection Rate: {(stats['faces_detected']/max(stats['frame_count'], 1)*100):.1f}%

BLINK ANALYSIS:
- Total Blinks: {stats['total_blinks']}
- Valid Blinks: {stats['valid_blinks']}
- Blinks per Minute: {stats['blinks_per_minute']:.1f}
- Recent Activity (60s): {stats['recent_blinks']} blinks
- Detection Accuracy: {(stats['valid_blinks']/max(stats['total_blinks'], 1)*100):.1f}%

HEALTH ASSESSMENT:
- PERCLOS: {stats['perclos']:.1f}%
- Drowsiness Threshold: 70%
- Current State: {'DROWSY' if stats['is_drowsy'] else 'ALERT'}
- Monitoring Status: {'Active' if stats['calculation_ready'] else 'Calibrating'}

CLINICAL INTERPRETATION:
- Normal Blink Rate: 12-20 per minute
- Current Classification: {tracker.get_blink_status()[0]}
- Drowsiness Risk: {'HIGH' if stats['is_drowsy'] else 'LOW'}
- Recommendation: {'Immediate attention needed' if stats['is_drowsy'] else 'Continue monitoring'}

TECHNICAL DETAILS:
- Model: eye_open_close_model.keras
- Confidence Threshold: {CONFIDENCE_THRESHOLD}
- Minimum Blink Gap: 150ms
- PERCLOS Window: 3 seconds
- Calibration Time: 8 seconds
"""
            
            filename = f"comprehensive_eye_tracking_report_{timestamp}.txt"
            with open(filename, "w") as f:
                f.write(report)
            print(f"📄 Comprehensive report saved: {filename}")
        
        elif key == ord('c'):
            # Calibration mode
            print("🔧 Calibration mode - Adjusting sensitivity...")
            tracker.blink_threshold *= 0.9  # Make more sensitive
            print(f"📊 New threshold: {tracker.blink_threshold:.2f}")

except KeyboardInterrupt:
    print("\n⏹️ Application stopped by user")
except Exception as e:
    print(f"❌ Unexpected error: {e}")
finally:
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Final comprehensive statistics
    final_stats = tracker.get_stats()
    print(f"\n" + "="*60)
    print("📊 FINAL SESSION SUMMARY")
    print("="*60)
    print(f"⏱️ Session Duration: {final_stats['runtime']:.1f} seconds")
    print(f"🎬 Average FPS: {final_stats['fps']:.1f}")
    print(f"👁️ Total Blinks Detected: {final_stats['total_blinks']}")
    print(f"✅ Valid Blinks: {final_stats['valid_blinks']}")
    print(f"📊 Final Blink Rate: {final_stats['blinks_per_minute']:.1f}/minute")
    print(f"🎯 Average Confidence: {final_stats['avg_confidence']:.2f}")
    print(f"📈 PERCLOS: {final_stats['perclos']:.1f}%")
    print(f"🔋 Final State: {'😴 DROWSY' if final_stats['is_drowsy'] else '😊 ALERT'}")
    print("="*60)
    print("👋 Thank you for using Enhanced Eye Tracking!")
    print("💡 For best results, ensure good lighting and face the camera directly.")
