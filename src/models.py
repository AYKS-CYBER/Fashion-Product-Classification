"""
models.py - Neural Network Architectures for Fashion Product Classification
Supports both Artificial Neural Network (ANN) and Convolutional Neural Network (CNN).
"""

import sys


def build_ann_model(input_shape=(28, 28, 1), num_classes=10, learning_rate=0.001):
    """
    Constructs and compiles an Artificial Neural Network (Multilayer Perceptron).

    Architecture:
        - Input / Flatten: (28, 28, 1) -> 784 features
        - Dense 256 + ReLU + BatchNorm + Dropout(0.3)
        - Dense 128 + ReLU + BatchNorm + Dropout(0.2)
        - Dense 64  + ReLU
        - Dense 10  + Softmax
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models, optimizers

    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Flatten(name="flatten_input"),
        
        layers.Dense(256, activation="relu", name="dense_1"),
        layers.BatchNormalization(name="bn_1"),
        layers.Dropout(0.30, name="dropout_1"),

        layers.Dense(128, activation="relu", name="dense_2"),
        layers.BatchNormalization(name="bn_2"),
        layers.Dropout(0.20, name="dropout_2"),

        layers.Dense(64, activation="relu", name="dense_3"),

        layers.Dense(num_classes, activation="softmax", name="output_probabilities")
    ], name="Fashion_ANN_Classifier")

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def build_cnn_model(input_shape=(28, 28, 1), num_classes=10, learning_rate=0.001):
    """
    Constructs and compiles a deep Convolutional Neural Network (CNN).

    Architecture:
        Conv Block 1:
            - Conv2D (32 filters, 3x3, same padding, ReLU)
            - BatchNormalization
            - Conv2D (32 filters, 3x3, same padding, ReLU)
            - MaxPooling2D (2x2)
            - Dropout (0.25)
        Conv Block 2:
            - Conv2D (64 filters, 3x3, same padding, ReLU)
            - BatchNormalization
            - Conv2D (64 filters, 3x3, same padding, ReLU)
            - MaxPooling2D (2x2)
            - Dropout (0.25)
        Fully Connected Head:
            - Flatten
            - Dense (128 units, ReLU)
            - BatchNormalization
            - Dropout (0.40)
            - Dense (10 units, Softmax)
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models, optimizers

    model = models.Sequential([
        layers.Input(shape=input_shape),

        # Conv Block 1
        layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv1_1"),
        layers.BatchNormalization(name="bn_c1_1"),
        layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv1_2"),
        layers.MaxPooling2D(pool_size=(2, 2), name="pool_1"),
        layers.Dropout(0.25, name="dropout_c1"),

        # Conv Block 2
        layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2_1"),
        layers.BatchNormalization(name="bn_c2_1"),
        layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2_2"),
        layers.MaxPooling2D(pool_size=(2, 2), name="pool_2"),
        layers.Dropout(0.25, name="dropout_c2"),

        # Classification Head
        layers.Flatten(name="flatten"),
        layers.Dense(128, activation="relu", name="dense_fc1"),
        layers.BatchNormalization(name="bn_fc1"),
        layers.Dropout(0.40, name="dropout_fc1"),
        layers.Dense(num_classes, activation="softmax", name="output_probabilities")
    ], name="Fashion_CNN_Classifier")

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def get_model(model_type="cnn", input_shape=(28, 28, 1), num_classes=10, learning_rate=0.001):
    """
    Factory function to instantiate either 'ann' or 'cnn'.
    """
    m_type = model_type.lower().strip()
    if m_type == "ann":
        return build_ann_model(input_shape=input_shape, num_classes=num_classes, learning_rate=learning_rate)
    elif m_type == "cnn":
        return build_cnn_model(input_shape=input_shape, num_classes=num_classes, learning_rate=learning_rate)
    else:
        raise ValueError(f"Unknown model type '{model_type}'. Choose 'ann' or 'cnn'.")


def print_architecture_comparison():
    """
    Prints a side-by-side conceptual comparison between ANN and CNN for image classification.
    """
    comparison = """
========================================================================================
                  ANN vs CNN Comparison for Fashion Product Classification
========================================================================================
Feature                    ANN (Multilayer Perceptron)          CNN (Convolutional Network)
----------------------------------------------------------------------------------------
Input Representation       1D Flattened vector (784 pixels)     2D Grid with channels (28x28x1)
Spatial Relationships      Lost (treats pixels independently)  Preserved via 2D convolutions
Feature Extraction         Learns global combinations only      Learns hierarchical local features
                                                                (edges -> textures -> shapes)
Translation Invariance     Low (shifted item looks different)   High (MaxPooling & convolution)
Parameter Efficiency       Higher parameter count per layer     Shared weights reduce redundancy
Benchmark Accuracy         ~88.0% - 89.5%                       ~92.0% - 94.0%
Inference Latency          Very fast (lightweight matrix mul)   Slightly higher computation
========================================================================================
"""
    print(comparison)


if __name__ == "__main__":
    print_architecture_comparison()
