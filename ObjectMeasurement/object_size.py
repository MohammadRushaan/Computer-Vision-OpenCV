# object_size.py
# ==============================================================================
# REAL-TIME & IMAGE OBJECT MEASUREMENT TOOL WITH HIGH-CONTRAST TEXT
# ==============================================================================

from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def draw_label(img, text, pos, bg_color=(20, 20, 20), text_color=(0, 255, 255)):
    """Draws text with a dark rounded background badge for 100% visibility."""
    x, y = int(pos[0]), int(pos[1])
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 2
    (w, h), baseline = cv2.getTextSize(text, font, scale, thickness)
    
    # Draw background box
    cv2.rectangle(img, (x - 4, y - h - 6), (x + w + 4, y + baseline), bg_color, -1)
    cv2.rectangle(img, (x - 4, y - h - 6), (x + w + 4, y + baseline), (255, 255, 255), 1)
    # Draw label text
    cv2.putText(img, text, (x, y - 2), font, scale, text_color, thickness, cv2.LINE_AA)

def process_frame(frame, ref_width, unit="cm"):
    """Finds objects, computes metric dimensions using the leftmost reference, and annotates."""
    output = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)

    # Edge detection with automatic thresholding based on median intensity
    v = np.median(blurred)
    lower = int(max(0, (1.0 - 0.33) * v))
    upper = int(min(255, (1.0 + 0.33) * v))
    edged = cv2.Canny(blurred, lower, upper)
    edged = cv2.dilate(edged, None, iterations=2)
    edged = cv2.erode(edged, None, iterations=1)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    if len(cnts) < 2:
        return output, "Need at least 2 distinct objects (reference + target)"

    # Sort contours strictly from left to right
    (cnts, _) = contours.sort_contours(cnts, method="left-to-right")
    pixelsPerMetric = None
    obj_count = 0

    for c in cnts:
        # Ignore minor shadows, dust, or noise specks
        if cv2.contourArea(c) < 600:
            continue

        box = cv2.minAreaRect(c)
        box = cv2.boxPoints(box) if hasattr(cv2, "boxPoints") else cv2.cv.BoxPoints(box)
        box = np.array(box, dtype="int")
        box = perspective.order_points(box)

        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)

        dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

        # First valid object on the left is ALWAYS our reference standard
        if pixelsPerMetric is None:
            pixelsPerMetric = dB / ref_width
            # Draw reference badge in Cyan
            cv2.drawContours(output, [box.astype("int")], -1, (255, 255, 0), 2)
            draw_label(output, f"REF ({ref_width:.1f}{unit})", (tl[0], tl[1] - 10), (100, 50, 0), (255, 255, 0))
            obj_count += 1
            continue

        dimA = dA / pixelsPerMetric
        dimB = dB / pixelsPerMetric

        # Draw detected object boundaries in Green
        cv2.drawContours(output, [box.astype("int")], -1, (0, 255, 0), 2)
        
        # Corner dots (Red) and midpoint dots (Blue)
        for (x, y) in box:
            cv2.circle(output, (int(x), int(y)), 4, (0, 0, 255), -1)
        for (mx, my) in [(tltrX, tltrY), (blbrX, blbrY), (tlblX, tlblY), (trbrX, trbrY)]:
            cv2.circle(output, (int(mx), int(my)), 4, (255, 0, 0), -1)

        # Dimension dimension axis lines (Magenta)
        cv2.line(output, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (255, 0, 255), 2)
        cv2.line(output, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255, 0, 255), 2)

        # Draw clear, shaded dimension badges
        draw_label(output, f"{dimA:.2f}{unit}", (tltrX - 20, tltrY - 15))
        draw_label(output, f"{dimB:.2f}{unit}", (trbrX + 10, trbrY))
        obj_count += 1

    return output, f"Tracking {obj_count} objects"

# ------------------------------------------------------------------------------
# EXECUTION ROUTE: IMAGE OR LIVE WEBCAM
# ------------------------------------------------------------------------------
ap = argparse.ArgumentParser(description="Object dimension scanner")
ap.add_argument("-i", "--image", help="Path to input image file (omit for webcam)")
ap.add_argument("-w", "--width", type=float, default=2.4, help="Real-world width of reference object (default: 2.4 cm for coin)")
ap.add_argument("-u", "--unit", default="cm", help="Measurement unit symbol (default: cm)")
args = vars(ap.parse_args())

if args.get("image", None) is not None:
    # Single image mode
    img = cv2.imread(args["image"])
    if img is None:
        raise FileNotFoundError(f"Could not open image: {args['image']}")
    img = imutils.resize(img, width=800)
    annotated, status = process_frame(img, args["width"], args["unit"])
    print(f"[INFO] {status}")
    cv2.imshow("Object Measurement Tool", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    # Live webcam mode
    print("[INFO] Starting Webcam. Place reference object on the FAR LEFT.")
    print("[INFO] Press 'q' in the window to quit.")
    vs = cv2.VideoCapture(0)
    
    while True:
        ret, frame = vs.read()
        if not ret:
            break
        frame = imutils.resize(frame, width=800)
        annotated, status = process_frame(frame, args["width"], args["unit"])
        
        cv2.putText(annotated, status, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow("Live Object Measurement Tool", annotated)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    vs.release()
    cv2.destroyAllWindows()