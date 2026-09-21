import cv2
import imutils

# ---------------------------------------------------------------------
# 1. Loading & Inspecting Dimensions
# ---------------------------------------------------------------------
image = cv2.imread("fruits.jpg")
(h, w, d) = image.shape
print(f"[INFO] Width: {w}, Height: {h}, Depth (Channels): {d}")

# ---------------------------------------------------------------------
# 2. Accessing Individual Pixels (BGR Order)
# ---------------------------------------------------------------------
# OpenCV uses BGR (Blue, Green, Red) instead of RGB
(B, G, R) = image[100, 50]
print(f"[INFO] Pixel at (x=50, y=100) - Red: {R}, Green: {G}, Blue: {B}")

# ---------------------------------------------------------------------
# 3. Array Slicing & Region of Interest (ROI / Cropping)
# ---------------------------------------------------------------------
# Crop format: image[startY:endY, startX:endX]
roi = image[60:260, 100:320]
cv2.imshow("Region of Interest (Cropped)", roi)
cv2.waitKey(0)

# ---------------------------------------------------------------------
# 4. Resizing Images
# ---------------------------------------------------------------------
# Manual aspect-ratio calculation:
# Target width = 300, compute proportional height
ratio = 300.0 / w
dim = (300, int(h * ratio))
resized_manual = cv2.resize(image, dim)

# Or easily via imutils (maintains aspect ratio automatically):
resized_auto = imutils.resize(image, width=300)
cv2.imshow("Resized", resized_auto)
cv2.waitKey(0)

# ---------------------------------------------------------------------
# 5. Rotating Images
# ---------------------------------------------------------------------
# Rotating manually requires computing an affine rotation matrix:
center = (w // 2, h // 2)
matrix = cv2.getRotationMatrix2D(center, angle=-45, scale=1.0)
rotated = cv2.warpAffine(image, matrix, (w, h))

# Or easily via imutils:
rotated_clean = imutils.rotate_bound(image, 45)
cv2.imshow("Rotated", rotated_clean)
cv2.waitKey(0)

# ---------------------------------------------------------------------
# 6. Smoothing / Blurring
# ---------------------------------------------------------------------
# Gaussian Blur reduces high-frequency sensor noise
# (11, 11) is the kernel size (must be odd numbers)
blurred = cv2.GaussianBlur(image, (11, 11), 0)
cv2.imshow("Blurred", blurred)
cv2.waitKey(0)

# ---------------------------------------------------------------------
# 7. Drawing Shapes and Text
# ---------------------------------------------------------------------
output = image.copy()

# Draw a red rectangle (top-left, bottom-right, BGR color, thickness)
cv2.rectangle(output, (50, 50), (250, 250), (0, 0, 255), 2)

# Draw a blue solid circle (center, radius, BGR color, -1 fills the circle)
cv2.circle(output, (150, 150), 20, (255, 0, 0), -1)

# Draw a green line (start_point, end_point, BGR color, thickness)
cv2.line(output, (20, 20), (280, 280), (0, 255, 0), 3)

# Put text over the image
cv2.putText(
    output,
    "OpenCV Basics",
    (10, 30),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (0, 255, 255),
    2,
)

cv2.imshow("Drawn Shapes & Text", output)
cv2.waitKey(0)
cv2.destroyAllWindows()