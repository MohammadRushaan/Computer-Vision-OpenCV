# Object_tracking.py
# ==============================================================================
# TRUE SPHERICAL BALL TRACKER: HOUGH GRADIENTS + COLOR + INERTIA CONFIRMATION
# ==============================================================================

from collections import deque
import argparse
import cv2
import imutils
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="Path to optional video file")
ap.add_argument("-b", "--buffer", type=int, default=32, help="Max contrail length")
args = vars(ap.parse_args())

# Maroon / Red HSV boundaries
redLower1 = np.array([0, 120, 30])
redUpper1 = np.array([10, 255, 190])
redLower2 = np.array([165, 120, 30])
redUpper2 = np.array([179, 255, 190])

pts = deque(maxlen=args["buffer"])
smoothed_center = None
SMOOTH_ALPHA = 0.6
MIN_MOVE_DIST = 3.5

vs = cv2.VideoCapture(0 if not args.get("video") else args["video"])

while True:
    ret, frame = vs.read()
    if frame is None:
        break

    frame = imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 1. Edge-preserving blur to assist gradient circle detection
    gray_blurred = cv2.medianBlur(gray, 7)
    
    # 2. Extract color mask for secondary verification
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, redLower1, redUpper1)
    mask2 = cv2.inRange(hsv, redLower2, redUpper2)
    color_mask = cv2.bitwise_or(mask1, mask2)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # --------------------------------------------------------------------------
    # 3. HOUGH CIRCLE TRANSFORM (GEOMETRIC SPHERICAL DETECTION)
    # --------------------------------------------------------------------------
    # cv2.HOUGH_GRADIENT computes edge directions and casts votes for circle centers.
    # dp=1.2: Resolution accumulator ratio
    # minDist=50: Minimum physical distance between two detected circles
    # param1=100: Higher threshold for the internal Canny edge detector
    # param2=32: Accumulator threshold (lower = more false circles, higher = strict circles)
    # minRadius/maxRadius: Absolute pixel bounds of typical balls
    detected_circles = cv2.HoughCircles(
        gray_blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=60,
        param1=100,
        param2=32,
        minRadius=15,
        maxRadius=90,
    )

    best_ball = None
    highest_confidence = 0

    if detected_circles is not None:
        detected_circles = np.uint16(np.around(detected_circles))

        for pt in detected_circles[0, :]:
            cx, cy, r = int(pt[0]), int(pt[1]), int(pt[2])

            # Ensure coordinates fit safely inside image boundaries
            if cy - r < 0 or cy + r >= frame.shape[0] or cx - r < 0 or cx + r >= frame.shape[1]:
                continue

            # ------------------------------------------------------------------
            # 4. CROSS-VERIFY: MEASURE COLOR CONCENTRATION INSIDE HOUGH CIRCLE
            # ------------------------------------------------------------------
            # Create a localized mask containing ONLY this geometric circle
            circle_mask = np.zeros(color_mask.shape, dtype="uint8")
            cv2.circle(circle_mask, (cx, cy), r, 255, -1)
            
            # Count color pixels inside this circle
            overlap = cv2.bitwise_and(color_mask, color_mask, mask=circle_mask)
            total_circle_pixels = np.pi * (r ** 2)
            color_fill_ratio = cv2.countNonZero(overlap) / float(total_circle_pixels)

            # Verification criteria:
            # 1. Hough found a sharp geometric circle
            # 2. More than 40% of its internal area matches the target ball color
            if color_fill_ratio > 0.40 and color_fill_ratio > highest_confidence:
                highest_confidence = color_fill_ratio
                best_ball = (cx, cy, r)

    # --------------------------------------------------------------------------
    # 5. SMOOTHING & MOTION-GATED CONTRAIL
    # --------------------------------------------------------------------------
    if best_ball is not None:
        bx, by, br = best_ball

        # Exponential Moving Average for position stabilization
        if smoothed_center is None:
            smoothed_center = (bx, by)
        else:
            sx = int(SMOOTH_ALPHA * bx + (1 - SMOOTH_ALPHA) * smoothed_center[0])
            sy = int(SMOOTH_ALPHA * by + (1 - SMOOTH_ALPHA) * smoothed_center[1])
            smoothed_center = (sx, sy)

        # Draw circle on detected spherical target
        cv2.circle(frame, (bx, by), br, (0, 255, 0), 2)
        cv2.circle(frame, smoothed_center, 4, (0, 0, 255), -1)

        # Append to contrail only if moving
        if len(pts) > 0 and pts[0] is not None:
            dist = np.hypot(smoothed_center[0] - pts[0][0], smoothed_center[1] - pts[0][1])
            if dist >= MIN_MOVE_DIST:
                pts.appendleft(smoothed_center)
            else:
                if len(pts) > 0:
                    pts.pop()
        else:
            pts.appendleft(smoothed_center)
    else:
        # Ball lost: smoothly clear trajectory
        if len(pts) > 0:
            pts.pop()

    # Draw trailing contrail
    for i in range(1, len(pts)):
        if pts[i - 1] is None or pts[i] is None:
            continue
        thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2)
        cv2.line(frame, pts[i - 1], pts[i], (0, 255, 255), max(1, thickness))

    cv2.imshow("Spherical Ball Tracker", frame)
    cv2.imshow("HSV Binary Mask", color_mask)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

vs.release()
cv2.destroyAllWindows()