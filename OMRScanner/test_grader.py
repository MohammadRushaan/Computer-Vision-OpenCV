# test_grader_pro.py
# ==============================================================================
# PRODUCTION-GRADE OPTICAL MARK RECOGNITION (OMR) BUBBLE SHEET GRADER
# Features:
# - Geometric perspective correction via four-point homography
# - Local adaptive binarization for uneven lighting and shadow resistance
# - Fill ratio analysis to catch blank questions and ambiguous double-bubbles
# - Detailed tabular console feedback and visual overlay
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. DEPENDENCY IMPORTS
# ------------------------------------------------------------------------------
import argparse
import cv2
import imutils
from imutils import contours
from imutils.perspective import four_point_transform
import numpy as np

# ------------------------------------------------------------------------------
# 2. CLI ARGUMENT PARSING
# ------------------------------------------------------------------------------
# Set up a command-line interface to allow passing different exam images easily.
ap = argparse.ArgumentParser(description="Advanced OMR Grader with Shadow & Ambiguity Handling")
ap.add_argument("-i", "--image", required=True, help="Path to the input exam photo/scan")
args = vars(ap.parse_args())

# ------------------------------------------------------------------------------
# 3. EXAM CONFIGURATION & SENSITIVITY THRESHOLDS
# ------------------------------------------------------------------------------
# Mapping: Question Index -> Correct Bubble Index (0=A, 1=B, 2=C, 3=D, 4=E)
ANSWER_KEY = {0: 1, 1: 4, 2: 0, 3: 3, 4: 1}
NUM_QUESTIONS = len(ANSWER_KEY)
CHOICES_PER_ROW = 5
TOTAL_BUBBLES = NUM_QUESTIONS * CHOICES_PER_ROW

# MIN_FILL_RATIO: At least 35% of the interior pixels must be filled for a mark to count.
# Any bubble with less than this is considered blank paper / an unshaded choice.
MIN_FILL_RATIO = 0.35

# AMBIGUOUS_DELTA: If the difference between the most-filled and second-most-filled
# bubble is less than 15%, mark the question as ambiguous (double-mark or incomplete erasure).
AMBIGUOUS_DELTA = 0.15

# ------------------------------------------------------------------------------
# 4. IMAGE LOADING & PREPROCESSING FOR DOCUMENT LOCALIZATION
# ------------------------------------------------------------------------------
image = cv2.imread(args["image"])
if image is None:
    raise FileNotFoundError(f"[ERROR] Could not load image from path: {args['image']}")

# Convert BGR color space to single-channel 8-bit grayscale (intensity range 0-255).
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply a 5x5 Gaussian blur to attenuate high-frequency pixel noise and subtle paper textures.
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Canny Edge Detector:
# Computes image gradients using Sobel operators, suppresses non-maximum points,
# and applies hysteresis thresholds (75 and 200) to trace prominent object silhouettes.
edged = cv2.Canny(blurred, 75, 200)

# ------------------------------------------------------------------------------
# 5. LOCATE THE 4 CORNERS OF THE EXAM PAGE
# ------------------------------------------------------------------------------
# Extract external contours from the binary edge map.
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
docCnt = None

if len(cnts) > 0:
    # Sort contours by surface area in descending order (largest polygon first).
    for c in sorted(cnts, key=cv2.contourArea, reverse=True):
        # Calculate the contour's closed perimeter length (arc length).
        peri = cv2.arcLength(c, True)
        
        # Douglas-Peucker Algorithm:
        # Simplifies a complex curved boundary into a polygon with fewer vertices.
        # Epsilon is set to 2% of the total perimeter.
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        
        # If the shape has exactly 4 vertices and covers at least 15% of the total image area,
        # we have found the page boundary.
        if len(approx) == 4 and cv2.contourArea(c) > (image.shape[0] * image.shape[1] * 0.15):
            docCnt = approx
            break

# ------------------------------------------------------------------------------
# 6. PERSPECTIVE RECTIFICATION (BIRD'S-EYE VIEW)
# ------------------------------------------------------------------------------
# If 4 corners were detected, compute the 3x3 homography matrix and warp the perspective.
# If no distinct page border exists (e.g. flatbed scanner or cropped photo), fall back to original canvas.
paper = four_point_transform(image, docCnt.reshape(4, 2)) if docCnt is not None else image.copy()
warped = four_point_transform(gray, docCnt.reshape(4, 2)) if docCnt is not None else gray.copy()

# ------------------------------------------------------------------------------
# 7. ADAPTIVE BINARIZATION (SHADOW & GRADIENT INVARIANCE)
# ------------------------------------------------------------------------------
# Unlike Otsu's global threshold, adaptive thresholding computes a local cutoff
# for each 25x25 pixel neighborhood using a Gaussian-weighted sum minus constant C (7).
# - cv2.THRESH_BINARY_INV: Inverts polarity so pencil marks/boundaries become WHITE (255)
#   and clean white paper background becomes BLACK (0).
thresh = cv2.adaptiveThreshold(
    warped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 7
)

# ------------------------------------------------------------------------------
# 8. LOCATE, FILTER, AND SORT BUBBLE CONTOURS
# ------------------------------------------------------------------------------
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
questionCnts = []

for c in cnts:
    (x, y, w, h) = cv2.boundingRect(c)
    ar = w / float(h)
    area = cv2.contourArea(c)
    
    # Structural filters to distinguish genuine answer bubbles from letters, noise, or lines:
    # 1. Width & Height between 15px and 80px (appropriate target scale).
    # 2. Aspect Ratio between 0.75 and 1.30 (roughly circular/square).
    # 3. Minimum contour area of 120 pixels (rejects punctuation and small specks).
    if 15 <= w <= 80 and 15 <= h <= 80 and 0.75 <= ar <= 1.30 and area >= 120:
        questionCnts.append(c)

# If artifacts passed the filters, keep only the top expected candidates sorted by area.
if len(questionCnts) > TOTAL_BUBBLES:
    questionCnts = sorted(questionCnts, key=cv2.contourArea, reverse=True)[:TOTAL_BUBBLES]

# Ensure we have all bubbles accounted for before proceeding to row processing.
if len(questionCnts) != TOTAL_BUBBLES:
    raise ValueError(
        f"[ERROR] Found {len(questionCnts)} candidate bubbles, but expected {TOTAL_BUBBLES}. "
        "Check lighting or adjust contour filters."
    )

# Sort all candidate bubbles spatially from top to bottom (Row 1 down to Row N).
questionCnts = contours.sort_contours(questionCnts, method="top-to-bottom")[0]

# ------------------------------------------------------------------------------
# 9. EVALUATE BUBBLE FILL RATIOS & GRADE QUESTIONS
# ------------------------------------------------------------------------------
correct = 0
results_log = []

# Step through detected bubbles in increments of CHOICES_PER_ROW (5 choices per question).
for (q, i) in enumerate(range(0, len(questionCnts), CHOICES_PER_ROW)):
    # Sort the 5 bubbles of the current row from left to right (Choice A through E).
    row_cnts = contours.sort_contours(
        questionCnts[i : i + CHOICES_PER_ROW], method="left-to-right"
    )[0]
    
    fill_ratios = []
    for c in row_cnts:
        # Create a single-channel black mask matching the paper dimensions.
        mask = np.zeros(thresh.shape, dtype="uint8")
        
        # Fill the interior of the candidate bubble contour with solid white (255).
        cv2.drawContours(mask, [c], -1, 255, -1)
        
        # Count total geometric pixels inside the contour boundary.
        bubble_area = cv2.countNonZero(mask)
        
        # Isolate filled pixels: bitwise AND keeps white pixels that lie inside our bubble mask.
        marked_mask = cv2.bitwise_and(thresh, thresh, mask=mask)
        marked_area = cv2.countNonZero(marked_mask)
        
        # Calculate the fill ratio: 0.0 = completely empty, 1.0 = completely filled.
        fill_ratios.append(marked_area / float(bubble_area))

    # Identify the indices of the highest and second-highest filled bubbles.
    sorted_indices = np.argsort(fill_ratios)[::-1]
    best_idx = sorted_indices[0]
    second_idx = sorted_indices[1]
    
    best_fill = fill_ratios[best_idx]
    second_fill = fill_ratios[second_idx]
    
    # Classify answer status:
    # 1. BLANK: Top fill is below the minimum threshold.
    # 2. DOUBLE_MARKED: Top two fills are very close, indicating erasure or multi-marking.
    # 3. VALID: One dominant, clearly filled bubble exists.
    if best_fill < MIN_FILL_RATIO:
        status = "BLANK"
        marked_idx = None
    elif (best_fill - second_fill) < AMBIGUOUS_DELTA and second_fill > MIN_FILL_RATIO:
        status = "DOUBLE_MARKED"
        marked_idx = None
    else:
        status = "VALID"
        marked_idx = best_idx

    # Grade the question against the ANSWER_KEY.
    ans_idx = ANSWER_KEY[q]
    
    if status == "VALID" and marked_idx == ans_idx:
        correct += 1
        # Draw a thick GREEN circle around the correct choice.
        cv2.drawContours(paper, [row_cnts[ans_idx]], -1, (0, 255, 0), 3)
        result = "CORRECT"
    else:
        # Highlight the correct answer in thin GREEN.
        cv2.drawContours(paper, [row_cnts[ans_idx]], -1, (0, 255, 0), 2)
        
        # If the student selected a specific wrong choice, highlight it in thick RED.
        if marked_idx is not None:
            cv2.drawContours(paper, [row_cnts[marked_idx]], -1, (0, 0, 255), 3)
        result = f"INCORRECT ({status})"

    # Log record for terminal printout.
    choice_chars = ["A", "B", "C", "D", "E"]
    student_ans = choice_chars[marked_idx] if marked_idx is not None else status
    results_log.append((q + 1, choice_chars[ans_idx], student_ans, result))

# ------------------------------------------------------------------------------
# 10. GENERATE TERMINAL REPORT & DISPLAY IMAGE
# ------------------------------------------------------------------------------
print("\n" + "=" * 50)
print(f"{'Q#':<4}{'KEY':<8}{'MARKED':<15}{'OUTCOME'}")
print("=" * 50)
for q_num, key, marked, outcome in results_log:
    print(f"{q_num:<4}{key:<8}{marked:<15}{outcome}")
print("=" * 50)

final_score = (correct / float(NUM_QUESTIONS)) * 100
print(f"Final Score: {final_score:.2f}% ({correct}/{NUM_QUESTIONS} Correct)\n")

# Render score box directly onto the graded paper.
cv2.rectangle(paper, (10, 10), (220, 60), (0, 0, 0), -1)
cv2.putText(
    paper,
    f"Score: {final_score:.1f}%",
    (20, 45),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.9,
    (255, 255, 255),
    2,
)

# Display the final graded document.
cv2.imshow("Production OMR Result", paper)
cv2.waitKey(0)
cv2.destroyAllWindows()