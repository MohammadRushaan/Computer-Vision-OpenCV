import cv2
import imutils

# 1. Load the input image
image = cv2.imread("tetris_blocks.png")
cv2.imshow("1. Original", image)
cv2.waitKey(0)

# 2. Convert to grayscale (simplifies 3 color channels down to 1 intensity channel)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("2. Grayscale", gray)
cv2.waitKey(0)

# 3. Edge detection using Canny algorithm
# Pixels with gradients between 30 and 150 are classified as edges
edges = cv2.Canny(gray, threshold1=30, threshold2=150)
cv2.imshow("3. Canny Edges", edges)
cv2.waitKey(0)

# 4. Thresholding: Convert grayscale to binary black & white
# Any pixel < 225 becomes 255 (white foreground), rest become 0 (black background)
thresh = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)[1]
cv2.imshow("4. Thresholded (Binary)", thresh)
cv2.waitKey(0)

# 5. Finding Contours (continuous curves along boundaries of white regions)
contours = cv2.findContours(
    thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)
contours = imutils.grab_contours(contours)

output = image.copy()
print(f"[INFO] Found {len(contours)} distinct object contours.")

# 6. Loop over detected contours and draw boundaries
for i, c in enumerate(contours):
  # Ignore tiny noise artifacts (e.g. area < 30 pixels)
  if cv2.contourArea(c) < 30:
    continue

  # Draw the contour outline in purple (thickness = 2)
  cv2.drawContours(output, [c], -1, (240, 0, 159), 2)

# Overlay total count text on top
cv2.putText(
    output,
    f"Objects: {len(contours)}",
    (10, 25),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (0, 255, 0),
    2,
)

cv2.imshow("5. Final Contours Detected", output)
cv2.waitKey(0)
cv2.destroyAllWindows()