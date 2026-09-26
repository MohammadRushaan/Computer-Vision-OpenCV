# detect_blinks.py
# ==============================================================================
# REAL-TIME EYE BLINK DETECTION USING OPENCV, DLIB, AND SCIPY
# Based on Adrian Rosebrock's PyImageSearch Tutorial (2017/04/24)
# Paper: "Real-Time Eye Blink Detection using Facial Landmarks" (Soukupová & Čech, 2016)
# ==============================================================================

from scipy.spatial import distance as dist
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2


def eye_aspect_ratio(eye):
    """
    Computes the Eye Aspect Ratio (EAR) as defined by Soukupová & Čech (2016).
    
    Formula:
             ||p2 - p6|| + ||p3 - p5||
       EAR = -------------------------
                    2 * ||p1 - p4||
                    
    - Numerator: Computes the average vertical distance between upper and lower eyelids.
    - Denominator: Computes the horizontal distance between outer and inner eye corners.
    - When the eye is open, EAR is roughly constant (~0.25 - 0.35).
    - When blinking, the eyelids touch, causing the numerator and EAR to rapidly approach zero.
    """
    # Compute the Euclidean distances between the two sets of vertical landmarks (p2, p6) and (p3, p5)
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # Compute the Euclidean distance between the horizontal eye landmark coordinates (p1, p4)
    C = dist.euclidean(eye[0], eye[3])

    # Compute and return the final Eye Aspect Ratio scalar
    ear = (A + B) / (2.0 * C)
    return ear


# ------------------------------------------------------------------------------
# 1. PARSE COMMAND LINE ARGUMENTS
# ------------------------------------------------------------------------------
ap = argparse.ArgumentParser(description="Eye blink detection using dlib 68-point landmarks")
ap.add_argument(
    "-p",
    "--shape-predictor",
    default="shape_predictor_68_face_landmarks.dat",
    help="Path to facial landmark predictor dat file",
)
ap.add_argument(
    "-v",
    "--video",
    type=str,
    default="",
    help="Optional path to input video file (leave empty to use live webcam)",
)
args = vars(ap.parse_args())

# ------------------------------------------------------------------------------
# 2. DEFINE SYSTEM THRESHOLDS & TRACKING STATE
# ------------------------------------------------------------------------------
# If the EAR falls below this threshold, the eye is considered closed
EYE_AR_THRESH = 0.23

# The eye must remain below the threshold for at least this many consecutive frames
# to register as a genuine blink (filters out camera noise and partial twitches)
EYE_AR_CONSEC_FRAMES = 3

# Frame counter tracking how long an eye has stayed below threshold
COUNTER = 0

# Total number of validated blinks detected
TOTAL = 0

# ------------------------------------------------------------------------------
# 3. INITIALIZE DLIB'S DETECTOR, PREDICTOR, AND LANDMARK SLICES
# ------------------------------------------------------------------------------
print("[INFO] Loading facial landmark predictor...")
# HOG + Linear SVM frontal face detector
detector = dlib.get_frontal_face_detector()
# Kazemi & Sullivan 68-point shape predictor
predictor = dlib.shape_predictor(args["shape_predictor"])

# Extract the 68-point landmark indices corresponding to the left and right eyes
# Left eye corresponds to points 42-47; Right eye corresponds to points 36-41
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# ------------------------------------------------------------------------------
# 4. INITIALIZE VIDEO STREAM
# ------------------------------------------------------------------------------
if args["video"] == "":
    print("[INFO] Starting live webcam stream. Press 'q' to quit...")
    vs = cv2.VideoCapture(0)
    # Pause briefly to allow camera auto-exposure to stabilize
    cv2.waitKey(500)
else:
    print(f"[INFO] Opening video file: {args['video']}")
    vs = cv2.VideoCapture(args["video"])

# ------------------------------------------------------------------------------
# 5. MAIN PROCESSING LOOP
# ------------------------------------------------------------------------------
while True:
    # Read the next frame from the camera or video file
    ret, frame = vs.read()
    if not ret or frame is None:
        break

    # Downscale the frame width to 650px to boost real-time processing FPS
    frame = imutils.resize(frame, width=650)

    # Convert the frame to grayscale for facial detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Windows-Safe Buffer Bridge: Enforce standard C-contiguous 8-bit array layout
    # to avoid dlib C++ pybind11 buffer errors
    gray_clean = np.ascontiguousarray(gray.copy(), dtype=np.uint8)

    # Detect faces in the grayscale frame (upsample factor 0 for optimal speed)
    rects = detector(gray_clean, 0)

    # Loop over all detected faces
    for rect in rects:
        # Determine the facial landmarks for the face region
        shape = predictor(gray_clean, rect)
        # Convert the dlib shape object into a standard (68, 2) NumPy integer array
        shape = face_utils.shape_to_np(shape)

        # Slice the landmark coordinates for the left and right eyes
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        # Calculate the Eye Aspect Ratio for both eyes
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # Average the two EAR values together for stability
        ear = (leftEAR + rightEAR) / 2.0

        # Compute the convex hull around each eye to outline them cleanly
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)

        # Draw the contours around both eyes on the display frame (Green)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # Check if the eye aspect ratio is below the blink threshold
        if ear < EYE_AR_THRESH:
            COUNTER += 1
        else:
            # If eyes were closed for at least EYE_AR_CONSEC_FRAMES, increment blink count
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                TOTAL += 1

            # Reset the closed frame counter
            COUNTER = 0

        # Display the live blink count and current EAR value on the video frame
        cv2.putText(
            frame,
            f"Blinks: {TOTAL}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
        cv2.putText(
            frame,
            f"EAR: {ear:.2f}",
            (300, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

    # Render the frame to the screen
    cv2.imshow("Real-Time Eye Blink Detection", frame)

    # Break out of the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Clean up video capture handles and close OpenCV GUI windows
vs.release()
cv2.destroyAllWindows()