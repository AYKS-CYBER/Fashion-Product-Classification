"""
dataset.py - Fashion-MNIST Dataset Loader and Preprocessing Pipeline
"""

import os
import numpy as np
from PIL import Image

# 10 categories in Fashion-MNIST
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
    "Ankle boot",
]

CLASS_DESCRIPTIONS = {
    0: "T-shirt or short-sleeved top",
    1: "Pants, trousers, or slacks",
    2: "Pullover, sweater, or hoodie",
    3: "Dress or one-piece gown",
    4: "Coat, jacket, or trench",
    5: "Open-toe sandal or flip-flop",
    6: "Button-up or collared shirt",
    7: "Sneaker or athletic running shoe",
    8: "Handbag, backpack, or tote",
    9: "Ankle-high boot",
}


def _generate_synthetic_fashion_data():
    """
    Fallback data generator for offline environments where dataset download is unavailable.
    Generates synthetic 28x28 grayscale images with geometric garment-like patterns.
    """
    np.random.seed(42)
    n_train, n_test = 2000, 500
    
    def make_samples(n):
        images = np.zeros((n, 28, 28), dtype=np.float32)
        labels = np.random.randint(0, 10, size=n)
        for i in range(n):
            cls = labels[i]
            # Create distinct patterns for different classes
            if cls == 0 or cls == 6:  # Shirt / T-shirt: rectangle torso + arms
                images[i, 4:24, 6:22] = 0.8
                images[i, 4:12, 2:6] = 0.6
                images[i, 4:12, 22:26] = 0.6
            elif cls == 1:  # Trouser: two columns
                images[i, 3:25, 7:12] = 0.85
                images[i, 3:25, 16:21] = 0.85
            elif cls == 2 or cls == 4:  # Pullover / Coat: wider body + sleeves
                images[i, 4:26, 5:23] = 0.75
                images[i, 6:22, 1:6] = 0.7
                images[i, 6:22, 22:27] = 0.7
            elif cls == 3:  # Dress: flared A-shape
                for r in range(4, 26):
                    w = int(3 + (r - 4) * 0.45)
                    images[i, r, 14-w:14+w] = 0.8
            elif cls == 5:  # Sandal: sole + straps
                images[i, 20:24, 4:24] = 0.9
                images[i, 12:15, 8:20] = 0.7
            elif cls == 7 or cls == 9:  # Sneaker / Boot: sole + upper body
                images[i, 18:24, 4:24] = 0.85
                images[i, 10:19, 4:16] = 0.8
                if cls == 9: # Boot is taller
                    images[i, 4:12, 4:14] = 0.75
            elif cls == 8:  # Bag: box + strap handle
                images[i, 10:24, 6:22] = 0.85
                images[i, 4:10, 10:18] = 0.5
                images[i, 6:10, 12:16] = 0.0  # open handle
            # Add subtle texture noise
            images[i] += np.random.normal(0, 0.05, (28, 28))
            images[i] = np.clip(images[i], 0.0, 1.0)
        return images, labels

    x_train, y_train = make_samples(n_train)
    x_test, y_test = make_samples(n_test)
    return (x_train, y_train), (x_test, y_test)


def load_fashion_mnist(val_split=0.15, flatten=False, random_state=42):
    """
    Loads and preprocesses the Fashion-MNIST dataset.

    Args:
        val_split (float): Fraction of training data to reserve for validation (default: 0.15).
        flatten (bool): If True, flattens (28, 28) into 784-dim vector for simple ANN.
                        If False, expands dimensions to (28, 28, 1) for CNN.
        random_state (int): Seed for reproducible train-validation split.

    Returns:
        tuple: (x_train, y_train), (x_val, y_val), (x_test, y_test)
    """
    loaded_from_tf = False
    try:
        import tensorflow as tf
        (x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = tf.keras.datasets.fashion_mnist.load_data()
        loaded_from_tf = True
        print("[dataset] Successfully loaded Fashion-MNIST dataset via TensorFlow Keras.")
    except Exception as e:
        print(f"[dataset] Notice: Standard TF download not available ({e}). Using robust fallback dataset.")
        (x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = _generate_synthetic_fashion_data()

    # Normalize pixel values to [0.0, 1.0] float32
    if loaded_from_tf:
        x_train = x_train_raw.astype("float32") / 255.0
        x_test = x_test_raw.astype("float32") / 255.0
    else:
        x_train = x_train_raw.astype("float32")
        x_test = x_test_raw.astype("float32")

    y_train = y_train_raw.astype("int64")
    y_test = y_test_raw.astype("int64")

    # Reshape images based on target model requirement
    if flatten:
        x_train = x_train.reshape((-1, 28 * 28))
        x_test = x_test.reshape((-1, 28 * 28))
    else:
        # (N, 28, 28, 1) for Convolutional Neural Network
        x_train = np.expand_dims(x_train, axis=-1)
        x_test = np.expand_dims(x_test, axis=-1)

    # Train-validation split
    num_samples = len(x_train)
    val_size = int(num_samples * val_split)

    np.random.seed(random_state)
    indices = np.random.permutation(num_samples)
    train_idx, val_idx = indices[val_size:], indices[:val_size]

    x_val, y_val = x_train[val_idx], y_train[val_idx]
    x_train, y_train = x_train[train_idx], y_train[train_idx]

    print(f"[dataset] Data loaded - Train: {x_train.shape}, Val: {x_val.shape}, Test: {x_test.shape}")
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def get_data_augmentation():
    """
    Returns a Keras data augmentation pipeline to reduce overfitting.
    """
    import tensorflow as tf
    from tensorflow.keras import layers

    return tf.keras.Sequential([
        layers.RandomRotation(0.04),
        layers.RandomTranslation(height_factor=0.04, width_factor=0.04),
        layers.RandomZoom(height_factor=(-0.04, 0.04)),
        layers.RandomFlip("horizontal"),
    ], name="data_augmentation")


def export_sample_images(output_dir="data/samples"):
    """
    Exports representative samples of each fashion category as PNG images
    so users can test single-image inference immediately.
    """
    os.makedirs(output_dir, exist_ok=True)
    try:
        (_, _), (_, _), (x_test, y_test) = load_fashion_mnist(val_split=0.1, flatten=False)
    except Exception:
        (x_test, y_test), _ = _generate_synthetic_fashion_data()
        x_test = np.expand_dims(x_test, -1)

    saved_paths = []
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        # Find first occurrence of this class
        matches = np.where(y_test == cls_idx)[0]
        if len(matches) > 0:
            sample_img = x_test[matches[0]]
            if sample_img.ndim == 3:
                sample_img = sample_img.squeeze(-1)
            # Scale back to 0-255 uint8
            img_uint8 = (sample_img * 255.0).astype(np.uint8)
            pil_img = Image.fromarray(img_uint8, mode="L")
            safe_name = cls_name.lower().replace("/", "_").replace(" ", "_")
            file_path = os.path.join(output_dir, f"sample_{safe_name}.png")
            pil_img.save(file_path)
            saved_paths.append((cls_name, file_path))

    print(f"[dataset] Exported {len(saved_paths)} sample category images to {output_dir}/")
    return saved_paths


if __name__ == "__main__":
    export_sample_images()
