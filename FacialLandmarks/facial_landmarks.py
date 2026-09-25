# facial_landmarks.py
# ==============================================================================
# ADVANCED FACIAL TELEMETRY HUD:
# - 68-Point Landmark Extraction
# - Real-Time Eye Aspect Ratio (EAR) & Blink Counter
# - Mouth Aspect Ratio (MAR) & Yawn Detection
# - Head Roll Angle Estimation
# ==============================================================================

from scipy.spatial import distance as dist
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2

# ------------------------------------------------------------------------------
# 1. MATHEMATICAL FORMULAS (EAR & MAR)
# ------------------------------------------------------------------------------
def eye_aspect_ratio(eye):
    """
    Computes the Eye Aspect Ratio (EAR) proposed by Soukupova & Cech (2016).
    Formula: EAR = (|p2 - p6| + |p3 - p5|) / (2 * |p1 - p4|)
    """
    # Compute the vertical distances between upper and lower eyelids
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # Compute horizontal distance between outer and inner eye corners
    C = dist.euclidean(eye[0], eye[3])

    # Return scalar aspect ratio (falls sharply when eyelid closes)
    return (A + B) / (2.0 * C)


def mouth_aspect_ratio(mouth):
    """
    Computes Mouth Aspect Ratio (MAR) using vertical and horizontal lip spans.
    """
    # Vertical distances between upper and lower lips
    A = dist.euclidean(mouth[2], mouth[10]) # 50 to 58
    B = dist.euclidean(mouth[4], mouth[8])   # 52 to 56

    # Horizontal distance between left and right mouth corners
    C = dist.euclidean(mouth[0], mouth[6])   # 48 to 54

    return (A + B) / (2.0 * C)


def calculate_head_roll(left_eye_center, right_eye_center):
    """
    Computes head roll (tilt angle in degrees) using the vector between eye centers.
    """
    dY = right_eye_center[1] - left_eye_center[1]
    dX = right_eye_center[0] - left_eye_center[0]
    angle = np.degrees(np.arctan2(dY, dX))
    return angle


# ------------------------------------------------------------------------------
# 2. THRESHOLDS & TRACKING CONSTANTS
# ------------------------------------------------------------------------------
EAR_THRESHOLD = 0.22      # Threshold below which eye is registered as closed
EAR_CONSEC_FRAMES = 2     # Consecutive frames to confirm a genuine blink (reject noise)

MAR_THRESHOLD = 0.65      # Threshold above which mouth is registered as open/yawn

# Telemetry counters
blink_counter = 0
total_blinks = 0
yawn_counter = 0
total_yawns = 0
is_yawning = False

# Visualization mode:
# 1 = Full Feature Mesh
# 2 = Raw Dots Only
# 3 = Eye & Mouth Tracking Only
display_mode = 1

# Extract landmark indices for eyes and mouth
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

# ------------------------------------------------------------------------------
# 3. INITIALIZE DLIB PREDICTORS
# ------------------------------------------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", default="shape_predictor_68_face_landmarks.dat",
                help="Path to predictor .dat weights")
args = vars(ap.parse_args())

print("[INFO] Loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

print("[INFO] Starting webcam stream...")
print("  [1] Toggle Full Landmark Mesh")
print("  [2] Toggle Raw Landmark Dots")
print("  [3] Minimal Eye/Mouth Focus")
print("  [R] Reset Counters")
print("  [Q] Quit")

vs = cv2.VideoCapture(0)
cv2.waitKey(500)

while True:
    ret, frame = vs.read()
    if not ret or frame is None:
        continue

    # Downscale frame width to maintain real-time FPS
    frame = imutils.resize(frame, width=750)
    h, w = frame.shape[:2]

    # Convert to grayscale for detector
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = np.ascontiguousarray(gray, dtype=np.uint8)

    # Detect faces
    rects = detector(gray, 0)

    for rect in rects:
        # Predict 68 coordinates
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # Slice relevant feature subsets
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        mouth = shape[mStart:mEnd]

        # Calculate EAR for both eyes and average them
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        # Calculate MAR
        mar = mouth_aspect_ratio(mouth)

        # Compute Eye Centers for Head Roll estimation
        leftEyeCenter = leftEye.mean(axis=0).astype("int")
        rightEyeCenter = rightEye.mean(axis=0).astype("int")
        roll_angle = calculate_head_roll(leftEyeCenter, rightEyeCenter)

        # ----------------------------------------------------------------------
        # BLINK LOGIC
        # ----------------------------------------------------------------------
        if ear < EAR_THRESHOLD:
            blink_counter += 1
        else:
            if blink_counter >= EAR_CONSEC_FRAMES:
                total_blinks += 1
            blink_counter = 0

        # ----------------------------------------------------------------------
        # YAWN / OPEN MOUTH LOGIC
        # ----------------------------------------------------------------------
        if mar > MAR_THRESHOLD:
            yawn_counter += 1
            if yawn_counter > 10 and not is_yawning:
                total_yawns += 1
                is_yawning = True
        else:
            yawn_counter = 0
            is_yawning = False

        # ----------------------------------------------------------------------
        # DRAWING VISUALIZATIONS ACCORDING TO DISPLAY MODE
        # ----------------------------------------------------------------------
        if display_mode == 1:
            # Mode 1: Draw anatomical contour lines
            for (name, (start, end)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
                pts = shape[start:end]
                if name in ["left_eye", "right_eye", "mouth"]:
                    cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 255), thickness=1)
                elif "jaw" in name:
                    cv2.polylines(frame, [pts], isClosed=False, color=(255, 200, 0), thickness=2)
                else:
                    cv2.polylines(frame, [pts], isClosed=False, color=(0, 255, 0), thickness=1)

        elif display_mode == 2:
            # Mode 2: Draw individual 68 landmark dots
            for (px, py) in shape:
                cv2.circle(frame, (px, py), 2, (0, 255, 0), -1)

        elif display_mode == 3:
            # Mode 3: Convex hulls over eyes and mouth only
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            mouthHull = cv2.convexHull(mouth)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [mouthHull], -1, (0, 0, 255), 1)

        # Draw line connecting eye centers (visualizing head tilt)
        cv2.line(frame, tuple(leftEyeCenter), tuple(rightEyeCenter), (255, 0, 255), 2)

        # Face bounding box
        (x, y, fw, fh) = face_utils.rect_to_bb(rect)
        cv2.rectangle(frame, (x, y), (x + fw, y + fh), (80, 80, 80), 1)

    # --------------------------------------------------------------------------
    # HUD TELEMETRY OVERLAY (TOP-LEFT BADGE)
    # --------------------------------------------------------------------------
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (280, 160), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.rectangle(frame, (10, 10), (280, 160), (200, 200, 200), 1)

    if len(rects) > 0:
        cv2.putText(frame, f"Blinks : {total_blinks}", (25, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"EAR    : {ear:.2f}", (25, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Yawns  : {total_yawns}", (25, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        cv2.putText(frame, f"MAR    : {mar:.2f}", (25, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Tilt   : {roll_angle:.1f} deg", (25, 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 100, 255), 2)
    else:
        cv2.putText(frame, "NO FACE DETECTED", (25, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Facial Landmarks & Telemetry HUD", frame)
    key = cv2.waitKey(1) & 0xFF

    # Keyboard controls
    if key == ord("1"):
        display_mode = 1
    elif key == ord("2"):
        display_mode = 2
    elif key == ord("3"):
        display_mode = 3
    elif key == ord("r"):
        total_blinks = 0
        total_yawns = 0
        print("[INFO] Counters reset.")
    elif key == ord("q"):
        break

vs.release()
cv2.destroyAllWindows()