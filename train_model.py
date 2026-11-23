"""
=============================================================================
Fashion Product Classification using Enhanced CNN in Python
=============================================================================
Author: Machine Learning College Project
Dataset: Fashion-MNIST (Zalando Research)
Description:
    Enhanced Convolutional Neural Network (CNN) featuring:
    - Data Augmentation (RandomFlip, RandomTranslation, RandomZoom)
    - VGG-style dual-convolution blocks (32, 64, 128 filters)
    - Batch Normalization after every convolution and dense layer
    - Dynamic learning rate reduction (ReduceLROnPlateau)
    - Higher precision on confusing classes (Shirt vs T-shirt vs Coat vs Pullover)

Fashion-MNIST 10 Class Labels Explained:
    Index | Label Name   | Description
    -------------------------------------------------------------------------
    0     | T-shirt/top  | Short-sleeved t-shirts, crewneck/v-neck tops
    1     | Trouser      | Pants, jeans, slacks, and trousers
    2     | Pullover     | Sweaters, sweatshirts, knitwear, and pullovers
    3     | Dress        | Women's one-piece dresses and gowns
    4     | Coat         | Outerwear, trench coats, winter jackets, overcoats
    5     | Sandal       | Open-toe sandals, flip-flops, and summer footwear
    6     | Shirt        | Button-up collared shirts and dress shirts
    7     | Sneaker      | Athletic shoes, running shoes, tennis trainers
    8     | Bag          | Handbags, backpacks, totes, and purses
    9     | Ankle boot   | High-top shoes, winter boots, and ankle boots
=============================================================================
"""

import os
import json
import tempfile

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from PIL import Image

# -----------------------------------------------------------------------------
# 1. Project Configuration & Directory Setup
# -----------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
SAMPLES_DIR = os.path.join(SCRIPT_DIR, "sample_images")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)

CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

CLASS_DESCRIPTIONS = {
    "T-shirt/top": "Casual everyday tops, t-shirts, and short-sleeved crewneck/v-neck shirts.",
    "Trouser": "Pants, trousers, jeans, and formal slacks.",
    "Pullover": "Sweaters, pullovers, long-sleeve knitwear, and sweatshirts.",
    "Dress": "Full-body dresses, party dresses, and sundresses.",
    "Coat": "Long jackets, trench coats, winter coats, and outerwear.",
    "Sandal": "Open-toe casual summer sandals and flat slip-ons.",
    "Shirt": "Button-down collared dress shirts with front buttons and collars.",
    "Sneaker": "Athletic footwear, running sneakers, and sports shoes.",
    "Bag": "Handbags, backpacks, tote bags, and luggage accessories.",
    "Ankle boot": "High-top ankle boots, leather boots, and winter footwear."
}


def load_and_preprocess_data():
    """
    Loads Fashion-MNIST dataset from Keras, normalizes pixel intensities,
    and reshapes dimensions for 2D Convolutional Neural Networks.
    """
    print("=" * 70)
    print("1. LOADING AND PREPROCESSING FASHION-MNIST DATASET")
    print("=" * 70)

    (X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()

    print(f"-> Raw Training Data Shape : {X_train.shape} (60,000 images of 28x28 pixels)")
    print(f"-> Raw Testing Data Shape  : {X_test.shape} (10,000 images of 28x28 pixels)")

    # Normalization: scale [0, 255] to [0.0, 1.0]
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0

    # Reshaping: add channel dimension (28, 28, 1)
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)

    print(f"-> Normalized & Reshaped Train Shape: {X_train.shape}")
    print(f"-> Normalized & Reshaped Test Shape : {X_test.shape}")

    return (X_train, y_train), (X_test, y_test)


def build_enhanced_cnn_model(input_shape=(28, 28, 1), num_classes=10):
    """
    Constructs an enhanced VGG-style deep Convolutional Neural Network
    with integrated data augmentation, dual-convolution blocks,
    and extensive batch normalization.
    """
    print("\n" + "=" * 70)
    print("2. BUILDING ENHANCED CONVOLUTIONAL NEURAL NETWORK (CNN)")
    print("=" * 70)

    inputs = layers.Input(shape=input_shape, name="input_image")

    # Data Augmentation (active during training only)
    # Improves generalization and stops the model from confusing Shirt vs T-shirt
    augmented = layers.RandomFlip("horizontal", name="aug_flip")(inputs)
    augmented = layers.RandomTranslation(0.04, 0.04, fill_mode="constant", fill_value=0.0, name="aug_trans")(augmented)
    augmented = layers.RandomZoom(0.04, fill_mode="constant", fill_value=0.0, name="aug_zoom")(augmented)

    # --- Conv Block 1 (32 Filters) ---
    x = layers.Conv2D(32, kernel_size=(3, 3), padding="same", activation="relu", name="conv1_1")(augmented)
    x = layers.BatchNormalization(name="bn1_1")(x)
    x = layers.Conv2D(32, kernel_size=(3, 3), padding="same", activation="relu", name="conv1_2")(x)
    x = layers.BatchNormalization(name="bn1_2")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool1")(x)
    x = layers.Dropout(0.20, name="dropout1")(x)

    # --- Conv Block 2 (64 Filters) ---
    x = layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu", name="conv2_1")(x)
    x = layers.BatchNormalization(name="bn2_1")(x)
    x = layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu", name="conv2_2")(x)
    x = layers.BatchNormalization(name="bn2_2")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool2")(x)
    x = layers.Dropout(0.25, name="dropout2")(x)

    # --- Conv Block 3 (128 Filters) ---
    x = layers.Conv2D(128, kernel_size=(3, 3), padding="same", activation="relu", name="conv3_1")(x)
    x = layers.BatchNormalization(name="bn3_1")(x)
    x = layers.Conv2D(128, kernel_size=(3, 3), padding="same", activation="relu", name="conv3_2")(x)
    x = layers.BatchNormalization(name="bn3_2")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool3")(x)
    x = layers.Dropout(0.30, name="dropout3")(x)

    # --- Classification Head ---
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(256, activation="relu", name="dense_features_1")(x)
    x = layers.BatchNormalization(name="bn_dense_1")(x)
    x = layers.Dropout(0.40, name="dropout_dense_1")(x)

    x = layers.Dense(128, activation="relu", name="dense_features_2")(x)
    x = layers.BatchNormalization(name="bn_dense_2")(x)
    x = layers.Dropout(0.30, name="dropout_dense_2")(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="output_probabilities")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="Fashion_MNIST_Enhanced_CNN")

    # Optimizer with Adam and initial learning rate 0.001
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()
    return model


def train_and_evaluate(model, X_train, y_train, X_test, y_test, epochs=15, batch_size=64):
    """
    Trains the enhanced CNN with learning rate scheduling for optimum convergence.
    """
    print("\n" + "=" * 70)
    print(f"3. TRAINING ENHANCED CNN ({epochs} EPOCHS, BATCH SIZE {batch_size})")
    print("=" * 70)

    # Learning rate scheduler: halves learning rate if validation loss plateaus
    lr_scheduler = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-5,
        verbose=1
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=[lr_scheduler],
        verbose=1
    )

    print("\n" + "=" * 70)
    print("4. EVALUATING MODEL ON UNSEEN TEST DATASET (10,000 IMAGES)")
    print("=" * 70)

    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)

    print(f"\n========================================================")
    print(f"  >>> FINAL TEST ACCURACY : {test_accuracy * 100:.2f}% <<<")
    print(f"  >>> FINAL TEST LOSS     : {test_loss:.4f} <<<")
    print(f"========================================================\n")

    return history, test_loss, test_accuracy


def plot_training_history(history, save_path):
    """Saves professional plots for accuracy and loss."""
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs_range = range(1, len(acc) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Enhanced Fashion-MNIST CNN - Training & Validation Metrics", fontsize=14, fontweight="bold")

    ax1.plot(epochs_range, [a * 100 for a in acc], "o-", label="Training Accuracy", color="#2563eb", linewidth=2)
    ax1.plot(epochs_range, [va * 100 for va in val_acc], "s--", label="Validation Accuracy", color="#16a34a", linewidth=2)
    ax1.set_title("Model Accuracy over Epochs", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Accuracy (%)", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="lower right")

    ax2.plot(epochs_range, loss, "o-", label="Training Loss", color="#dc2626", linewidth=2)
    ax2.plot(epochs_range, val_loss, "s--", label="Validation Loss", color="#ea580c", linewidth=2)
    ax2.set_title("Model Loss over Epochs", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Cross-Entropy Loss", fontsize=11)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"-> Training history plot saved to: {save_path}")


def save_artifacts(model, history, test_loss, test_accuracy):
    """Saves the trained model and metadata."""
    print("\n" + "=" * 70)
    print("5. SAVING ENHANCED MODEL AND ARTIFACTS")
    print("=" * 70)

    keras_model_path = os.path.join(MODELS_DIR, "fashion_cnn_model.keras")
    model.save(keras_model_path)
    print(f"-> Model saved in native format: {keras_model_path}")

    h5_model_path = os.path.join(MODELS_DIR, "fashion_cnn_model.h5")
    try:
        model.save(h5_model_path)
        print(f"-> Model saved in H5 format: {h5_model_path}")
    except Exception as e:
        print(f"   (H5 format note: {e})")

    metadata = {
        "project_name": "Fashion Product Classification using CNN",
        "dataset": "Fashion-MNIST",
        "input_shape": [28, 28, 1],
        "num_classes": len(CLASS_NAMES),
        "class_names": CLASS_NAMES,
        "class_descriptions": CLASS_DESCRIPTIONS,
        "test_accuracy": float(test_accuracy),
        "test_loss": float(test_loss),
        "total_parameters": int(model.count_params()),
        "model_file": "fashion_cnn_model.keras",
        "model_type": "Enhanced VGG-style CNN with Data Augmentation",
        "preprocessing": {
            "grayscale": True,
            "target_size": [28, 28],
            "normalization": "x / 255.0",
            "background": "dark (white object on dark background)"
        }
    }

    metadata_path = os.path.join(MODELS_DIR, "model_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print(f"-> Metadata and class labels saved to: {metadata_path}")

    plot_path = os.path.join(MODELS_DIR, "training_history.png")
    plot_training_history(history, plot_path)


def main():
    print("""
    ==================================================================
       Fashion Product Classification - Enhanced CNN Pipeline
       VGG-Style Architecture + Data Augmentation
    ==================================================================
    """)

    (X_train, y_train), (X_test, y_test) = load_and_preprocess_data()
    model = build_enhanced_cnn_model(input_shape=(28, 28, 1), num_classes=10)
    history, test_loss, test_accuracy = train_and_evaluate(
        model, X_train, y_train, X_test, y_test, epochs=15, batch_size=64
    )
    save_artifacts(model, history, test_loss, test_accuracy)

    print("\n" + "=" * 70)
    print("SUCCESS: Enhanced model trained and saved successfully!")
    print(f"Trained Model Location: {os.path.join(MODELS_DIR, 'fashion_cnn_model.keras')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
