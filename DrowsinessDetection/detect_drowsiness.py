# detect_drowsiness.py
# ==============================================================================
# ADVANCED DRIVER DROWSINESS & FATIGUE DETECTION SYSTEM
# - Continuous Asynchronous Acoustic Alarm (winsound loop)
# - Dual EAR (Eye Closure) & MAR (Yawn) Monitoring
# - Rolling PERCLOS (Percentage of Eye Closure) Fatigue Index
# - Flashing Red Screen Hazard Warning HUD
# ==============================================================================

from scipy.spatial import distance as dist
from collections import deque
from imutils import face_utils
import threading
import argparse
import winsound
import imutils
import dlib
import cv2
import time
import numpy as np


# ------------------------------------------------------------------------------
# 1. CONTINUOUS THREADED AUDIO ALARM
# ------------------------------------------------------------------------------
class ContinuousAlarm:
    """
    Manages a background daemon thread that sounds an alarm continuously 
    at rapid intervals until explicitly stopped.
    """
    def __init__(self, frequency=2500, beep_duration=250, pause_duration=0.05):
        self.frequency = frequency
        self.beep_duration = beep_duration
        self.pause_duration = pause_duration
        self.is_ringing = False
        self.thread = None
        self._lock = threading.Lock()

    def _loop(self):
        while True:
            with self._lock:
                if not self.is_ringing:
                    break
            # Play high-intensity alert tone
            winsound.Beep(self.frequency, self.beep_duration)
            time.sleep(self.pause_duration)

    def start(self):
        with self._lock:
            if not self.is_ringing:
                self.is_ringing = True
                self.thread = threading.Thread(target=self._loop, daemon=True)
                self.thread.start()

    def stop(self):
        with self._lock:
            self.is_ringing = False


# ------------------------------------------------------------------------------
# 2. MATHEMATICAL ASPECT RATIO HELPERS
# ------------------------------------------------------------------------------
def eye_aspect_ratio(eye):
    """
    Computes Eye Aspect Ratio (EAR) from Soukupova and Cech (2016).
    EAR = (|p2 - p6| + |p3 - p5|) / (2 * |p1 - p4|)
    """
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)


def mouth_aspect_ratio(mouth):
    """
    Computes Mouth Aspect Ratio (MAR) to monitor sustained yawning.
    """
    A = dist.euclidean(mouth[2], mouth[10])  # Upper/lower inner lip height
    B = dist.euclidean(mouth[4], mouth[8])    # Upper/lower outer lip height
    C = dist.euclidean(mouth[0], mouth[6])    # Lip corners width
    return (A + B) / (2.0 * C)


# ------------------------------------------------------------------------------
# 3. CLI ARGUMENTS & HYPERPARAMETERS
# ------------------------------------------------------------------------------
ap = argparse.ArgumentParser(description="Real-Time Driver Fatigue Monitoring")
ap.add_argument("-p", "--shape-predictor", default="shape_predictor_68_face_landmarks.dat",
                help="Path to shape predictor weights")
ap.add_argument("-w", "--webcam", type=int, default=0, help="Webcam device index")
args = vars(ap.parse_args())

# Detection Thresholds
EAR_THRESH = 0.24           # Below this EAR, eyes are considered closed
EYE_CLOSED_CONSEC = 35      # Consecutive frames (~1.2 sec) before triggering alarm
MAR_THRESH = 0.70           # Above this MAR, mouth is wide open (yawn)
YAWN_CONSEC = 30            # Consecutive frames mouth is open before triggering yawn alert

# State variables
eye_closed_frames = 0
mouth_open_frames = 0
total_yawns = 0
yawn_active = False

# Rolling buffer for calculating PERCLOS (% of eye closure over last 100 frames)
closure_history = deque(maxlen=100)

alarm = ContinuousAlarm(frequency=2800, beep_duration=200, pause_duration=0.04)

# ------------------------------------------------------------------------------
# 4. INITIALIZE MODELS & VIDEO
# ------------------------------------------------------------------------------
print("[INFO] Loading 68-point facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# Extract landmark indices
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

print("[INFO] Starting webcam stream. Press 'q' to quit.")
vs = cv2.VideoCapture(args["webcam"])
cv2.waitKey(500)

frame_count = 0

try:
    while True:
        ret, frame = vs.read()
        if not ret or frame is None:
            break

        frame_count += 1
        frame = imutils.resize(frame, width=720)
        h, w = frame.shape[:2]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_clean = np.ascontiguousarray(gray.copy(), dtype=np.uint8)

        rects = detector(gray_clean, 0)
        is_drowsy = False

        if len(rects) > 0:
            rect = rects[0]  # Focus primary driver
            shape = predictor(gray_clean, rect)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            mouth = shape[mStart:mEnd]

            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            mar = mouth_aspect_ratio(mouth)

            # Draw visual contours
            cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [cv2.convexHull(mouth)], -1, (255, 200, 0), 1)

            # Record eye closure for rolling PERCLOS
            is_closed = ear < EAR_THRESH
            closure_history.append(1 if is_closed else 0)

            # ------------------------------------------------------------------
            # DROWSINESS LOGIC (EYES CLOSED)
            # ------------------------------------------------------------------
            if is_closed:
                eye_closed_frames += 1
                if eye_closed_frames >= EYE_CLOSED_CONSEC:
                    is_drowsy = True
                    alarm.start()
            else:
                eye_closed_frames = 0
                if not yawn_active:
                    alarm.stop()

            # ------------------------------------------------------------------
            # YAWN LOGIC
            # ------------------------------------------------------------------
            if mar > MAR_THRESH:
                mouth_open_frames += 1
                if mouth_open_frames >= YAWN_CONSEC and not yawn_active:
                    total_yawns += 1
                    yawn_active = True
            else:
                mouth_open_frames = 0
                yawn_active = False

            # Calculate PERCLOS percentage
            perclos = (sum(closure_history) / len(closure_history)) * 100 if len(closure_history) > 0 else 0

            # ------------------------------------------------------------------
            # HUD TELEMETRY BADGE
            # ------------------------------------------------------------------
            hud = frame.copy()
            cv2.rectangle(hud, (10, 10), (320, 155), (15, 15, 15), -1)
            cv2.addWeighted(hud, 0.75, frame, 0.25, 0, frame)
            cv2.rectangle(frame, (10, 10), (320, 155), (180, 180, 180), 1)

            ear_color = (0, 0, 255) if ear < EAR_THRESH else (0, 255, 0)
            cv2.putText(frame, f"EAR : {ear:.2f} (Thresh: {EAR_THRESH})", (20, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, ear_color, 2)
            cv2.putText(frame, f"MAR : {mar:.2f} (Thresh: {MAR_THRESH})", (20, 68),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"Yawns Total : {total_yawns}", (20, 98),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 180, 0), 2)
            cv2.putText(frame, f"PERCLOS     : {perclos:.1f}%", (20, 128),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

            # ------------------------------------------------------------------
            # FLASHING WARNING OVERLAY ON ALERT
            # ------------------------------------------------------------------
            if is_drowsy:
                # Flash every 4 frames
                if (frame_count // 4) % 2 == 0:
                    # Draw thick flashing red border
                    cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 14)

                    # Red warning banner
                    banner = frame.copy()
                    cv2.rectangle(banner, (w // 2 - 250, h // 2 - 50), (w // 2 + 250, h // 2 + 50), (0, 0, 255), -1)
                    cv2.addWeighted(banner, 0.85, frame, 0.15, 0, frame)
                    cv2.putText(frame, "DROWSINESS DETECTED!", (w // 2 - 220, h // 2 + 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3)

        else:
            # Silence alarm if face disappears from view
            alarm.stop()
            cv2.putText(frame, "NO DRIVER DETECTED", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Driver Drowsiness Alert System", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    # Ensure background beep thread safely terminates on exit
    alarm.stop()
    vs.release()
    cv2.destroyAllWindows()