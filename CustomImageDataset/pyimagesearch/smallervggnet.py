# pyimagesearch/smallervggnet.py

# ==============================================================================
# 1. IMPORT REQUIRED TENSORFLOW / KERAS MODULES
# ==============================================================================
# Sequential allows us to stack layers linearly from input to output
from tensorflow.keras.models import Sequential

# Convolutional, pooling, and regularization layers
from tensorflow.keras.layers import (
    Conv2D,               # 2D spatial convolution for feature extraction
    MaxPooling2D,         # Downsampling layer to reduce spatial dimensions
    Activation,           # Applies non-linear activation functions (ReLU, Softmax)
    Flatten,              # Flattens 2D feature maps into a 1D feature vector
    Dropout,              # Drops random activations to prevent overfitting
    Dense,                # Fully connected classification layer
    BatchNormalization    # Normalizes layer inputs to accelerate convergence
)

# Backend to inspect tensor channel order (channels_first vs channels_last)
from tensorflow.keras import backend as K


# ==============================================================================
# 2. DEFINE THE SMALLERVGGNET CLASS
# ==============================================================================
class SmallerVGGNet:
    @staticmethod
    def build(width, height, depth, classes):
        """
        Builds the SmallerVGGNet architecture.
        - width: Spatial width of input image (96)
        - height: Spatial height of input image (96)
        - depth: Color channels (3 for RGB)
        - classes: Total number of prediction categories (e.g. 3 for 3 Pokémon)
        """
        # Initialize an empty sequential model
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1  # Default to channels-last (TensorFlow default)

        # Handle channels-first configuration if necessary
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        # ======================================================================
        # BLOCK 1: (CONV => RELU => BN) => POOL => DROPOUT
        # ======================================================================
        # 32 filters of size (3, 3) to extract early low-level features
        model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        
        # Max-pool with (3, 3) pool size reduces 96x96 down to 32x32
        model.add(MaxPooling2D(pool_size=(3, 3)))
        model.add(Dropout(0.25))

        # ======================================================================
        # BLOCK 2: (CONV => RELU => BN) * 2 => POOL => DROPOUT
        # ======================================================================
        # Stacking two 3x3 convolutions increases receptive field before downsampling
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        
        # Downsamples from 32x32 to 16x16
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # ======================================================================
        # BLOCK 3: (CONV => RELU => BN) * 2 => POOL => DROPOUT
        # ======================================================================
        # Increase depth to 128 filters to learn complex textures and object shapes
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        
        # Downsamples from 16x16 to 8x8
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # ======================================================================
        # BLOCK 4: FLATTEN => DENSE => RELU => BN => DROPOUT
        # ======================================================================
        # Flatten 8x8x128 into a 1D vector (8192 elements)
        model.add(Flatten())
        
        # Fully connected layer with 1024 neurons
        model.add(Dense(1024))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # ======================================================================
        # BLOCK 5: OUTPUT CLASSIFICATION HEAD
        # ======================================================================
        # Output layer with one neuron per class
        model.add(Dense(classes))
        
        # Softmax turns raw logits into probability distribution summing to 1.0
        model.add(Activation("softmax"))

        return model