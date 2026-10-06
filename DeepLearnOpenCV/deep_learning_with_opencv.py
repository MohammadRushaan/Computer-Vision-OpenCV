# deep_learning_with_opencv.py

# ==============================================================================
# 1. IMPORT REQUIRED PACKAGES
# ==============================================================================
import numpy as np       # Numerical computing: array operations, index sorting, and probability indexing
import argparse          # Command-line interface parser for receiving script arguments
import time              # Benchmark inference latency (forward-pass execution time)
import cv2               # OpenCV: handles image loading, blob preprocessing, DNN model execution, and GUI display

# ==============================================================================
# 2. CONSTRUCT AND PARSE COMMAND-LINE ARGUMENTS
# ==============================================================================
# Initialize argument parser
ap = argparse.ArgumentParser(description="Image Classification using OpenCV DNN module")

# Path to the input target image that will be classified
ap.add_argument("-i", "--image", required=True, 
                help="Path to the input image")

# Path to the Caffe deploy configuration file describing layer architectures
ap.add_argument("-p", "--prototxt", required=True, 
                help="Path to Caffe 'deploy' prototxt file")

# Path to the pre-trained weights binary (.caffemodel file)
ap.add_argument("-m", "--model", required=True, 
                help="Path to pre-trained Caffe model weights")

# Path to the plain text file containing ImageNet category names/synsets
ap.add_argument("-l", "--labels", required=True, 
                help="Path to ImageNet synset labels file")

# Convert arguments into an accessible dictionary
args = vars(ap.parse_args())

# ==============================================================================
# 3. LOAD AND VERIFY INPUT DATA
# ==============================================================================
# Read the target image from disk using OpenCV (stored in BGR channel layout)
image = cv2.imread(args["image"])

# Verify that the image exists and loaded properly
if image is None:
    raise SystemExit(f"[ERROR] Could not load image from: {args['image']}")

# Open the synset label file and parse line-by-line:
# - .read().strip().split("\n") creates a list of individual lines
# - Each row looks like: 'n01440764 tench, Tinca tinca'
# - r.find(" ") identifies the separation after the synset ID code
# - .split(",")[0] keeps only the primary human-friendly name (e.g., 'tench')
rows = open(args["labels"]).read().strip().split("\n")
classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]

# ==============================================================================
# 4. PREPROCESS INPUT IMAGE INTO A 4D BLOB
# ==============================================================================
# Deep neural networks require a fixed spatial dimension and normalized channel inputs.
# cv2.dnn.blobFromImage executes:
# 1. Resizing to (224, 224) - standard input dimensions for SqueezeNet and GoogLeNet
# 2. Mean channel subtraction using (104.0, 117.0, 123.0) for B, G, and R channels
#    (centers pixel distribution around 0 based on ImageNet dataset averages)
# 3. Shape transformation into standard NCHW layout: (1, 3, 224, 224)
blob = cv2.dnn.blobFromImage(
    image, 
    scalefactor=1.0, 
    size=(224, 224), 
    mean=(104.0, 117.0, 123.0)
)

# ==============================================================================
# 5. LOAD CAFFE MODEL & RUN FORWARD INFERENCE
# ==============================================================================
print("[INFO] Loading Caffe deep learning model...")
# Initialize the network using the network definition (.prototxt) and learned parameters (.caffemodel)
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# Assign the preprocessed blob as the input to the network's initial layer
net.setInput(blob)

# Track the exact start timestamp
start = time.time()

# Run forward propagation through all layers to compute class probabilities
preds = net.forward()

# Track finish timestamp and print total inference latency
end = time.time()
print(f"[INFO] Classification took {end - start:.5f} seconds")

# ==============================================================================
# 6. EXTRACT AND DISPLAY TOP-5 PREDICTIONS
# ==============================================================================
# Flatten output probabilities in case the tensor returns extra dimensions
preds = preds.flatten()

# Apply softmax normalization if the output layer returns unnormalized logits
if preds.max() > 1.0 or preds.min() < 0.0:
    exp_preds = np.exp(preds - np.max(preds))
    preds = exp_preds / np.sum(exp_preds)

# Sort prediction indices in descending order (highest probability first)
# np.argsort sorts ascending; [::-1] reverses it; [:5] keeps the top 5
idxs = np.argsort(preds)[::-1][:5]

# Iterate through the top 5 predicted categories
for (i, idx) in enumerate(idxs):
    label = classes[idx]
    confidence = preds[idx] * 100

    # Overlay only the #1 top prediction text onto the image
    if i == 0:
        text = f"Label: {label}, {confidence:.2f}%"
        # Draw red text (BGR: 0, 0, 255) at pixel coordinates (x=10, y=30)
        cv2.putText(
            image, 
            text, 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (0, 0, 255), 
            2
        )
    
    # Print ranked results with percentage confidence to the console
    print(f"[INFO] {i + 1}. Label: {label} (Probability: {preds[idx]:.5f})")

# ==============================================================================
# 7. DISPLAY RESULT WINDOW
# ==============================================================================
# Display annotated image in an OpenCV GUI window
cv2.imshow("Classification Result", image)

# Wait indefinitely until any keyboard key is pressed inside the GUI window
cv2.waitKey(0)

# Release display resources and close window
cv2.destroyAllWindows()