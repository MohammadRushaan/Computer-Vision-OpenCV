# live_measure.py
# ==============================================================================
# CALIBRATED REAL-TIME MEASUREMENT TOOL WITH TEMPORAL SMOOTHING & FREEZE MODE
# ==============================================================================

from scipy.spatial import distance as dist
from imutils import perspective
import numpy as np
import argparse
import cv2

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def draw_label(img, text, pos, bg_color=(20, 20, 20), text_color=(0, 255, 255)):
    x, y = int(pos[0]), int(pos[1])
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 2
    (w, h), baseline = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(img, (x - 3, y - h - 5), (x + w + 3, y + baseline), bg_color, -1)
    cv2.rectangle(img, (x - 3, y - h - 5), (x + w + 3, y + baseline), (255, 255, 255), 1)
    cv2.putText(img, text, (x, y - 2), font, scale, text_color, thickness, cv2.LINE_AA)

ap = argparse.ArgumentParser()
ap.add_argument("-w", "--width", type=float, default=8.56, help="Reference width in cm (default: 8.56 for card)")
args = vars(ap.parse_args())

REF_WIDTH = args["width"]
pixelsPerMetric = None
frozen = False
frozen_frame = None

# EMA smoothing state variables
smooth_dimA = None
smooth_dimB = None
ALPHA = 0.35  # Smoothing factor (lower = smoother, higher = more responsive)

vs = cv2.VideoCapture(0)

print("\n" + "=" * 60)
print("FINAL OMR / MEASUREMENT HUD INSTRUCTIONS:")
print(" [C]       : Calibrate scale using card inside the central target box")
print(" [SPACE]   : Freeze / Unfreeze current measurement snapshot")
print(" [R]       : Reset calibration")
print(" [Q]       : Quit")
print("=" * 60 + "\n")

while True:
    if not frozen:
        ret, frame = vs.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        box_w, box_h = 420, 320
        x1 = (w - box_w) // 2
        y1 = (h - box_h) // 2
        x2 = x1 + box_w
        y2 = y1 + box_h

        roi = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 6
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

        cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        measured_box = None
        dA_pixels = 0
        dB_pixels = 0

        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(c) > 2000:
                box = cv2.minAreaRect(c)
                box = cv2.boxPoints(box)
                box = np.array(box, dtype="int")
                box = perspective.order_points(box)
                measured_box = box

                (tl, tr, br, bl) = box
                (tltrX, tltrY) = midpoint(tl, tr)
                (blbrX, blbrY) = midpoint(bl, br)
                (tlblX, tlblY) = midpoint(tl, bl)
                (trbrX, trbrY) = midpoint(tr, br)

                dA_pixels = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
                dB_pixels = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

                cv2.drawContours(roi, [box.astype("int")], -1, (0, 255, 0), 2)
                cv2.line(roi, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (255, 0, 255), 2)
                cv2.line(roi, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255, 0, 255), 2)

                if pixelsPerMetric is not None:
                    raw_dimA = dA_pixels / pixelsPerMetric
                    raw_dimB = dB_pixels / pixelsPerMetric

                    # Apply EMA temporal filter to stabilize readouts
                    if smooth_dimA is None:
                        smooth_dimA = raw_dimA
                        smooth_dimB = raw_dimB
                    else:
                        smooth_dimA = (ALPHA * raw_dimA) + ((1.0 - ALPHA) * smooth_dimA)
                        smooth_dimB = (ALPHA * raw_dimB) + ((1.0 - ALPHA) * smooth_dimB)

                    draw_label(roi, f"H: {smooth_dimA:.1f}cm", (tltrX - 25, tltrY - 10))
                    draw_label(roi, f"W: {smooth_dimB:.1f}cm", (trbrX + 5, trbrY))

        roi_color = (0, 255, 0) if pixelsPerMetric is not None else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), roi_color, 2)

        # UI Diagnostic Overlay
        if pixelsPerMetric is None:
            cv2.putText(frame, "STATUS: UNCALIBRATED", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "Hold card flat in box & press 'C'", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        else:
            cv2.putText(frame, f"STATUS: CALIBRATED ({pixelsPerMetric:.1f} px/cm)", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Press SPACE to freeze reading | 'R' to reset", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        display_image = frame
    else:
        display_image = frozen_frame

    cv2.imshow("Webcam Object Measurement Tool", display_image)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("c") and not frozen:
        if measured_box is not None and max(dA_pixels, dB_pixels) > 0:
            card_pixel_extent = max(dA_pixels, dB_pixels)
            pixelsPerMetric = card_pixel_extent / REF_WIDTH
            smooth_dimA = None
            smooth_dimB = None
            print(f"[SUCCESS] Calibrated: {pixelsPerMetric:.2f} px/cm")

    elif key == 32:  # Spacebar toggles freeze mode
        frozen = not frozen
        if frozen:
            frozen_frame = frame.copy()
            cv2.putText(frozen_frame, "[FROZEN SNAPSHOT]", (w - 240, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    elif key == ord("r"):
        pixelsPerMetric = None
        smooth_dimA = None
        smooth_dimB = None
        frozen = False
        print("[INFO] Calibration reset.")

    elif key == ord("q"):
        break

vs.release()
cv2.destroyAllWindows()