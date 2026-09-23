import os
import cv2
import imutils
import numpy as np
from skimage.filters import threshold_local

# ==============================================================================
# SECTION 1: MATHEMATICAL & GEOMETRIC COORDINATE HELPERS
# ==============================================================================

def order_points(pts):
    """
    Sorts 4 arbitrary (x, y) coordinates into deterministic clockwise order:
    [0: Top-Left, 1: Top-Right, 2: Bottom-Right, 3: Bottom-Left].
    
    Mathematical Principles:
    - Origin (0,0) is at top-left of image plane.
    - Sum (x + y): Top-left is closest to origin -> MINIMUM sum.
                   Bottom-right is furthest from origin -> MAXIMUM sum.
    - Difference (y - x): Top-right has small y, large x -> MINIMUM difference.
                          Bottom-left has large y, small x -> MAXIMUM difference.
    """
    rect = np.zeros((4, 2), dtype="float32")

    # Compute coordinate sums (axis 1 sums columns across each row)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-Left
    rect[2] = pts[np.argmax(s)]  # Bottom-Right

    # Compute coordinate differences (y - x via np.diff)
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-Right
    rect[3] = pts[np.argmax(diff)]  # Bottom-Left

    return rect


def four_point_transform(image, pts):
    """
    Performs a 4-point planar perspective transformation (Homography) to warp 
    a distorted quadrilateral into a flat, top-down rectangular scan.
    """
    # Step A: Enforce coordinate ordering
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Step B: Determine maximum width using Euclidean Distance:
    # d = sqrt((x2 - x1)^2 + (y2 - y1)^2)
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # Step C: Determine maximum height using Euclidean Distance
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Guard: Prevent matrix singular errors if zero area is passed
    if maxWidth <= 0 or maxHeight <= 0:
        return None

    # Step D: Construct destination coordinates matching target dimensions
    dst = np.array(
        [
            [0, 0],                         # Dest Top-Left
            [maxWidth - 1, 0],              # Dest Top-Right
            [maxWidth - 1, maxHeight - 1],  # Dest Bottom-Right
            [0, maxHeight - 1],             # Dest Bottom-Left
        ],
        dtype="float32",
    )

    # Step E: Calculate the 3x3 perspective transformation matrix M
    M = cv2.getPerspectiveTransform(rect, dst)

    # Step F: Apply geometric warp transformation
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


# ==============================================================================
# SECTION 2: ADAPTIVE COMPUTER VISION DETECTION PIPELINE
# ==============================================================================

def detect_target_document(resized_frame):
    """
    Multi-pass document boundary detector:
    1. HSV Saturation Channel: Isolates colored documents on light/white desks.
    2. Morphological Gradient: Detects boundaries despite low overall contrast.
    3. Convex Hull: Eliminates border notches caused by holding fingers.
    4. Area Clamping: Rejects entire-frame edges (>95% frame size).
    """
    h, w = resized_frame.shape[:2]
    total_area = h * w

    # Pass 1: Saturation Masking (HSV Space)
    # White desks/paper have near-zero saturation; colored slips show high saturation
    hsv = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2HSV)
    _, s_channel, _ = cv2.split(hsv)
    s_blur = cv2.GaussianBlur(s_channel, (7, 7), 0)
    _, s_thresh = cv2.threshold(s_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Pass 2: Grayscale Morphological Gradient
    # Highlights intensity boundaries independent of absolute brightness
    gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
    g_blur = cv2.GaussianBlur(gray, (7, 7), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph_grad = cv2.morphologyEx(g_blur, cv2.MORPH_GRADIENT, kernel)
    g_thresh = cv2.adaptiveThreshold(
        morph_grad, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2
    )

    # Test candidate masks sequentially
    for mask in [s_thresh, g_thresh]:
        # Morphological Closing: Bridges gaps in broken boundaries
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

        cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        # Discard spurious background borders (>95% area) or small desk clutter (<8% area)
        valid_cnts = [
            c for c in cnts 
            if (total_area * 0.95) > cv2.contourArea(c) > (total_area * 0.08)
        ]
        valid_cnts = sorted(valid_cnts, key=cv2.contourArea, reverse=True)

        for c in valid_cnts:
            # Convex Hull: Creates a taut bounding polygon, bridging across fingers
            hull = cv2.convexHull(c)
            peri = cv2.arcLength(hull, True)
            
            # Douglas-Peucker polygon approximation
            approx = cv2.approxPolyDP(hull, 0.03 * peri, True)
            if len(approx) == 4:
                return approx
                
            # Fallback: Minimum rotated bounding box if edges remain rounded
            rect_box = cv2.minAreaRect(hull)
            box_pts = cv2.boxPoints(rect_box)
            return np.int32(box_pts).reshape((4, 1, 2))

    return None


# ==============================================================================
# SECTION 3: INTERACTIVE USER OVERRIDE & MOUSE HANDLING
# ==============================================================================

manual_points = []

def click_event(event, x, y, flags, param):
    """Mouse event callback to record exact user corner selections."""
    global manual_points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(manual_points) < 4:
            manual_points.append((x, y))
            cv2.circle(param, (x, y), 6, (0, 0, 255), -1)
            cv2.putText(
                param,
                str(len(manual_points)),
                (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Click 4 Corners of Document", param)


def pick_corners_manually(preview_img):
    """Enables manual 4-point selection when automatic detection fails or requires correction."""
    global manual_points
    manual_points = []
    clone = preview_img.copy()

    cv2.imshow("Click 4 Corners of Document", clone)
    cv2.setMouseCallback("Click 4 Corners of Document", click_event, clone)

    print("\n[Manual Corner Mode]")
    print("- Click 4 corners of document: Top-Left, Top-Right, Bottom-Right, Bottom-Left.")
    print("- Press 'Enter' or 'Space' to confirm points.")
    print("- Press 'r' to reset points.")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key in (13, 32):  # Enter or Space
            if len(manual_points) == 4:
                break
            else:
                print(f"Select all 4 corners (currently: {len(manual_points)}).")
        elif key == ord("r"):
            manual_points = []
            clone = preview_img.copy()
            cv2.imshow("Click 4 Corners of Document", clone)
            cv2.setMouseCallback("Click 4 Corners of Document", click_event, clone)
            print("Points reset.")

    cv2.destroyAllWindows()
    return np.array(manual_points, dtype="float32")


# ==============================================================================
# SECTION 4: POST-PROCESSING, THRESHOLDING & EXPORT
# ==============================================================================

def process_and_save_scan(orig_image, corners_scaled):
    """Warps document, computes adaptive local threshold, and saves outputs."""
    print("Processing scan...")
    warped = four_point_transform(orig_image, corners_scaled)

    if warped is None:
        print("Error: Perspective transformation failed.")
        return

    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    # Adaptive block size: prevents the all-white washout bug by scaling
    # neighborhood evaluation relative to final image resolution
    block_size = int(warped.shape[0] / 25)
    if block_size % 2 == 0:
        block_size += 1
    block_size = max(15, block_size)

    # Local threshold surface generation
    T = threshold_local(warped_gray, block_size, offset=10, method="gaussian")
    warped_bin = (warped_gray > T).astype("uint8") * 255

    # Save to disk
    cv2.imwrite("scanned_output_color.jpg", warped)
    cv2.imwrite("scanned_output_bw.jpg", warped_bin)
    print("Export Complete: 'scanned_output_color.jpg' and 'scanned_output_bw.jpg'")

    # Display side-by-side verification
    preview_color = imutils.resize(warped, height=600)
    preview_bw = imutils.resize(warped_bin, height=600)
    preview_bw_bgr = cv2.cvtColor(preview_bw, cv2.COLOR_GRAY2BGR)

    combined = np.hstack([preview_color, preview_bw_bgr])
    cv2.imshow("Color Scan (Left) | Cleaned B&W (Right)", combined)
    print("Press any key to close the preview.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ==============================================================================
# SECTION 5: APPLICATION ENTRY POINT & ROUTING
# ==============================================================================

print("=== Robust OpenCV Document Scanner ===")
print("Select Input Mode:")
print("  [1] Live Camera Feed (Stabilized EMA)")
print("  [2] Load Local Image File (Auto + Manual Override)")
choice = input("Enter choice (1 or 2): ").strip()

# MODE 1: LIVE WEBCAM WITH EXPONENTIAL MOVING AVERAGE (EMA)
if choice == "1":
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot access webcam.")
        exit()

    print("\n[Live Camera Active]")
    print("- Press 's' to capture and scan.")
    print("- Press 'q' to quit.")

    smoothed_corners = None
    alpha = 0.35  # EMA weight: higher = responsive, lower = smooth

    captured_orig = None
    captured_contour = None
    captured_ratio = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        orig = frame.copy()
        ratio = frame.shape[0] / 500.0
        resized = imutils.resize(frame, height=500)

        detected_approx = detect_target_document(resized)
        display_frame = resized.copy()

        if detected_approx is not None:
            # Sort current detection points
            current_ordered = order_points(
                detected_approx.reshape(4, 2)
            ).astype("float32")

            # Temporal Exponential Smoothing: S_t = alpha * X_t + (1 - alpha) * S_{t-1}
            # Eliminates camera coordinate jitter
            if smoothed_corners is None:
                smoothed_corners = current_ordered
            else:
                smoothed_corners = (
                    alpha * current_ordered + (1.0 - alpha) * smoothed_corners
                )

            stable_contour = smoothed_corners.astype("int32").reshape((4, 1, 2))
            cv2.drawContours(display_frame, [stable_contour], -1, (0, 255, 0), 2)
            cv2.putText(
                display_frame,
                "Document Locked! Press 's' to Scan",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
        else:
            smoothed_corners = None
            stable_contour = None
            cv2.putText(
                display_frame,
                "Detecting document boundaries...",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

        cv2.imshow("Document Scanner (Live Feed)", display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            if stable_contour is not None:
                captured_orig = orig
                captured_contour = stable_contour
                captured_ratio = ratio
                break
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if captured_contour is not None:
        scaled_corners = captured_contour.reshape(4, 2) * captured_ratio
        process_and_save_scan(captured_orig, scaled_corners)

# MODE 2: LOAD IMAGE FILE
elif choice == "2":
    image_path = input("Enter image path: ").strip().strip('"')
    if not os.path.exists(image_path):
        print(f"Error: Path '{image_path}' not found.")
        exit()

    image = cv2.imread(image_path)
    if image is None:
        print("Error: OpenCV cannot read the target image file.")
        exit()

    ratio = image.shape[0] / 500.0
    orig = image.copy()
    resized = imutils.resize(image, height=500)

    contour = detect_target_document(resized)

    if contour is not None:
        preview = resized.copy()
        cv2.drawContours(preview, [contour], -1, (0, 255, 0), 2)
        cv2.putText(
            preview,
            "SPACE: Accept | M: Manually select corners",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
        )
        cv2.imshow("Detection Preview", preview)
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()

        if key in (ord("m"), ord("M")):
            manual_pts = pick_corners_manually(resized)
            corners_scaled = manual_pts * ratio
        else:
            corners_scaled = contour.reshape(4, 2) * ratio
    else:
        print("Auto-detection missed. Launching manual corner selector...")
        manual_pts = pick_corners_manually(resized)
        corners_scaled = manual_pts * ratio

    process_and_save_scan(orig, corners_scaled)

else:
    print("Invalid option selected. Exiting.")