"""
=============================================================================
Fashion Product Classifier - Streamlit Web Application
=============================================================================
A modern interactive web interface for classifying fashion products
into 10 categories using an Enhanced Convolutional Neural Network (CNN)
trained on the Fashion-MNIST dataset.

Features:
- Clean, professional dark/light high-contrast UI (No emojis, academic look)
- Smart aspect-ratio preserving letterboxing (prevents squishing shirts into coats)
- Automatic background detection & inversion (handles real-world white background photos)
- Visual preprocessing pipeline inspector
- Full probability distribution across all 10 classes

Author: Machine Learning College Project
Framework: Streamlit, TensorFlow/Keras, NumPy, Pillow
=============================================================================
"""

import os
import json
import numpy as np
import streamlit as st
from PIL import Image, ImageOps, ImageEnhance

# -----------------------------------------------------------------------------
# 1. Page Configuration & Styling (High Contrast, Professional UI)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Fashion Product Classifier | Enhanced CNN",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS - High Contrast on both Dark and Light Themes
st.markdown("""
<style>
    /* Main Header */
    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: #ffffff;
        margin-bottom: 0.3rem;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: inline-block;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-bottom: 1.4rem;
        line-height: 1.5;
    }
    
    /* Clean Cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 12px;
        padding: 1.3rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
    }
    .card-label {
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        color: #38bdf8;
        margin-bottom: 6px;
    }
    .prediction-badge {
        font-size: 1.8rem;
        font-weight: 800;
        color: #ffffff;
        padding: 0.5rem 1rem;
        background: rgba(37, 99, 235, 0.25);
        border: 1px solid rgba(96, 165, 250, 0.4);
        border-left: 5px solid #38bdf8;
        border-radius: 8px;
        display: inline-block;
        margin: 6px 0 10px 0;
    }
    .confidence-text {
        color: #cbd5e1;
        font-size: 1.05rem;
    }
    .confidence-val {
        color: #38bdf8;
        font-weight: 800;
        font-size: 1.25rem;
    }
    .desc-text {
        color: #94a3b8;
        font-size: 0.92rem;
        margin-top: 8px;
        line-height: 1.5;
    }
    
    /* Callouts */
    .info-callout {
        background: rgba(30, 41, 59, 0.65);
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-left: 4px solid #38bdf8;
        padding: 1rem 1.2rem;
        border-radius: 8px;
        color: #e2e8f0;
        font-size: 0.94rem;
        line-height: 1.6;
        margin-bottom: 1.4rem;
    }
    .info-callout strong {
        color: #38bdf8;
    }
    
    .tip-callout {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-left: 4px solid #10b981;
        padding: 0.9rem 1.1rem;
        border-radius: 8px;
        color: #cbd5e1;
        font-size: 0.91rem;
        line-height: 1.6;
        margin-top: 1rem;
    }
    .tip-callout strong {
        color: #34d399;
    }
    
    /* Empty State Box */
    .empty-state-box {
        border: 2px dashed rgba(255, 255, 255, 0.18);
        border-radius: 12px;
        padding: 3rem 2rem;
        text-align: center;
        background: rgba(30, 41, 59, 0.3);
        color: #94a3b8;
    }
    .empty-state-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #e2e8f0;
        margin-bottom: 8px;
    }
    
    /* Step Headers */
    .step-header {
        font-size: 1.25rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .step-badge {
        background: #2563eb;
        color: #ffffff;
        font-size: 0.8rem;
        font-weight: 800;
        padding: 2px 8px;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. Paths and Class Constants
# -----------------------------------------------------------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(APP_DIR, "models")
MODEL_KERAS_PATH = os.path.join(MODELS_DIR, "fashion_cnn_model.keras")
MODEL_H5_PATH = os.path.join(MODELS_DIR, "fashion_cnn_model.h5")
METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")
SAMPLES_DIR = os.path.join(APP_DIR, "sample_images")

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
    "T-shirt/top": "Casual everyday tops, crewneck and v-neck t-shirts without buttons or formal collars.",
    "Trouser": "Pants, trousers, denim jeans, and formal slacks.",
    "Pullover": "Sweaters, pullovers, long-sleeve knitwear, and closed-front sweatshirts.",
    "Dress": "Women's full-body one-piece dresses, party dresses, and gowns.",
    "Coat": "Long jackets, winter coats, trench coats, and outerwear with wide lapels.",
    "Sandal": "Open-toe footwear, flip-flops, and summer flat sandals.",
    "Shirt": "Button-down collared dress shirts with front buttons and distinct collars.",
    "Sneaker": "Athletic footwear, running sneakers, and sports trainers.",
    "Bag": "Handbags, backpacks, tote bags, and luggage accessories.",
    "Ankle boot": "High-top boots, winter leather boots, and ankle-height footwear."
}


# -----------------------------------------------------------------------------
# 3. Model & Metadata Loading
# -----------------------------------------------------------------------------
@st.cache_resource
def load_trained_model():
    """Loads the trained CNN model from the models folder."""
    import tensorflow as tf
    from tensorflow import keras

    if os.path.exists(MODEL_KERAS_PATH):
        try:
            model = keras.models.load_model(MODEL_KERAS_PATH)
            return model, MODEL_KERAS_PATH
        except Exception as e:
            st.error(f"Error loading Keras format: {e}")

    if os.path.exists(MODEL_H5_PATH):
        try:
            model = keras.models.load_model(MODEL_H5_PATH)
            return model, MODEL_H5_PATH
        except Exception as e:
            st.error(f"Error loading H5 format: {e}")

    return None, None


@st.cache_data
def load_metadata():
    """Loads saved model metadata."""
    if os.path.exists(METADATA_PATH):
        try:
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


# -----------------------------------------------------------------------------
# 4. Smart Preprocessing Function (Crucial for Real-World Accuracy)
# -----------------------------------------------------------------------------
def smart_preprocess_image(
    pil_img,
    background_mode="Auto-detect",
    preserve_aspect_ratio=True,
    contrast_factor=1.2,
    clean_background=True
):
    """
    Preprocesses real-world photos to match the Fashion-MNIST domain:
    1. Grayscale Conversion: Converts RGB (3 channels) into single channel.
    2. Smart Inversion: Detects whether background is light (real photos) or dark (Fashion-MNIST).
       Inverts if background is light so clothing is bright on dark background.
    3. Aspect-Ratio Preserving Letterboxing:
       Instead of squishing a 3:4 portrait into a square (which flattens shirts into coats),
       scales the item to fit within a ~24x24 box and centers it on a 28x28 black canvas.
    4. Background Noise Suppression:
       Removes soft gray shadows/table reflections, clamping near-black background to pure 0.
    5. Normalization: Scales pixels to [0.0, 1.0].
    """
    # Step 1: Grayscale
    gray = pil_img.convert("L")

    # Step 2: Background Inversion Decision
    arr = np.array(gray)
    h, w = arr.shape
    border_samples = [
        arr[0, 0], arr[0, -1], arr[-1, 0], arr[-1, -1],
        arr[0, w // 2], arr[-1, w // 2], arr[h // 2, 0], arr[h // 2, -1]
    ]
    border_mean = float(np.mean(border_samples))
    is_detected_light_bg = border_mean > 120.0

    if background_mode == "White Background (Invert to Dark)":
        invert_needed = True
    elif background_mode == "Dark Background (Keep As-Is)":
        invert_needed = False
    else:  # Auto-detect
        invert_needed = is_detected_light_bg

    if invert_needed:
        gray = ImageOps.invert(gray)
        arr = np.array(gray)

    # Step 3: Contrast Enhancement (sharpens collars and buttons)
    if contrast_factor != 1.0:
        enhancer = ImageEnhance.Contrast(gray)
        gray = enhancer.enhance(contrast_factor)
        arr = np.array(gray)

    # Step 4: Background Noise Suppression
    if clean_background:
        bg_noise = float(np.percentile(arr, 15))
        if bg_noise > 15.0:
            arr_float = arr.astype(np.float32)
            arr_float = np.clip((arr_float - bg_noise) * (255.0 / max(1.0, 255.0 - bg_noise)), 0, 255)
            gray = Image.fromarray(arr_float.astype(np.uint8))

    # Step 5: Resize to 28x28
    if preserve_aspect_ratio:
        target_size = 24
        gray.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)
        canvas = Image.new("L", (28, 28), color=0)
        offset_x = (28 - gray.width) // 2
        offset_y = (28 - gray.height) // 2
        canvas.paste(gray, (offset_x, offset_y))
        final_img = canvas
    else:
        final_img = gray.resize((28, 28), Image.Resampling.LANCZOS)

    # Step 6: Normalize to [0.0, 1.0] and reshape to (1, 28, 28, 1)
    img_array = np.array(final_img, dtype=np.float32) / 255.0
    input_tensor = np.expand_dims(img_array, axis=(0, -1))

    return input_tensor, final_img, invert_needed


# -----------------------------------------------------------------------------
# 5. Sidebar: Project Controls & Presets
# -----------------------------------------------------------------------------
st.sidebar.markdown("### Project Configuration")

model, model_path_used = load_trained_model()
metadata = load_metadata()

if model is not None:
    st.sidebar.success("Model Status: Online")
    if metadata and "test_accuracy" in metadata:
        st.sidebar.metric(
            label="Evaluation Accuracy",
            value=f"{metadata['test_accuracy'] * 100:.2f}%"
        )
        st.sidebar.caption(f"Trainable Parameters: `{metadata.get('total_parameters', 'N/A'):,}`")
else:
    st.sidebar.error("Model Not Found")
    st.sidebar.markdown("""
    **To train the model, run in terminal:**
    ```bash
    python train_model.py
    ```
    """)

st.sidebar.markdown("---")
st.sidebar.subheader("Quick-Test Samples")
st.sidebar.write("Test instantly with official Fashion-MNIST samples:")

sample_options = ["None (Upload custom image)"]
sample_files = []
if os.path.exists(SAMPLES_DIR):
    sample_files = [f for f in os.listdir(SAMPLES_DIR) if f.endswith(".png")]
    for f in sample_files:
        clean_name = f.replace("sample_", "").replace(".png", "").replace("_", " ").title()
        sample_options.append(clean_name)

selected_sample_name = st.sidebar.selectbox("Select Benchmark Sample:", sample_options)

st.sidebar.markdown("---")
st.sidebar.subheader("Preprocessing Controls")
bg_mode = st.sidebar.selectbox(
    "Background Inversion:",
    ["Auto-detect", "White Background (Invert to Dark)", "Dark Background (Keep As-Is)"],
    help="Real photos typically have white backgrounds and need to be inverted to dark."
)

preserve_ratio = st.sidebar.checkbox(
    "Preserve Aspect Ratio (Letterbox)",
    value=True,
    help="Recommended! Prevents squishing rectangular photos (e.g., shirts) into coats."
)

clean_bg = st.sidebar.checkbox(
    "Clean Background Noise",
    value=True,
    help="Clamps soft shadows/reflections to pure black (matching Fashion-MNIST)."
)

contrast_val = st.sidebar.slider(
    "Contrast Boost:",
    min_value=0.8,
    max_value=2.0,
    value=1.2,
    step=0.1,
    help="Higher contrast emphasizes collar and button lines for Shirts vs T-shirts."
)

st.sidebar.markdown("---")
st.sidebar.subheader("10 Fashion Classes")
for idx, name in enumerate(CLASS_NAMES):
    st.sidebar.caption(f"**{idx}.** {name}")

# -----------------------------------------------------------------------------
# 6. Main Application Body
# -----------------------------------------------------------------------------
st.markdown('<div class="hero-title">Fashion Product Classifier</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Deep Learning Computer Vision system using an Enhanced Convolutional Neural Network (CNN) '
    'trained on the Fashion-MNIST benchmark dataset.</div>',
    unsafe_allow_html=True
)

if model is None:
    st.warning("""
    **Trained Model Not Found inside `models/` folder.**
    
    Please run the training pipeline first:
    ```powershell
    python train_model.py
    ```
    """)
    st.stop()

# Clean Info Callout (Without LaTeX formatting errors)
st.markdown("""
<div class="info-callout">
    <strong>Technical Note on Upper-Body Classification:</strong>
    In Fashion-MNIST, items are standardized <strong>28 × 28 pixel</strong> silhouettes.
    Our application utilizes <strong>Smart Aspect-Ratio Letterboxing</strong> and <strong>Automatic Background Inversion</strong>
    so that collars, sleeves, and buttons are preserved accurately and shirts are not distorted into wide coats.
</div>
""", unsafe_allow_html=True)

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown('<div class="step-header"><span class="step-badge">STEP 1</span> Select or Upload Product Image</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload a clothing photo (JPG, JPEG, or PNG):",
        type=["jpg", "jpeg", "png"],
        help="Upload a photo of a shirt, t-shirt, shoe, bag, trouser, dress, etc."
    )

    active_image = None
    image_source_info = ""

    if uploaded_file is not None:
        try:
            active_image = Image.open(uploaded_file)
            image_source_info = f"Uploaded File: `{uploaded_file.name}`"
        except Exception as e:
            st.error(f"Error opening image: {e}")
    elif selected_sample_name != "None (Upload custom image)":
        target_file = None
        for f in sample_files:
            clean_name = f.replace("sample_", "").replace(".png", "").replace("_", " ").title()
            if clean_name == selected_sample_name:
                target_file = f
                break
        if target_file:
            sample_path = os.path.join(SAMPLES_DIR, target_file)
            active_image = Image.open(sample_path)
            image_source_info = f"Fashion-MNIST Test Sample: `{selected_sample_name}`"

    if active_image is not None:
        st.caption(image_source_info)

        # Preprocess the image using the smart pipeline
        input_tensor, preview_28x28, was_inverted = smart_preprocess_image(
            active_image,
            background_mode=bg_mode,
            preserve_aspect_ratio=preserve_ratio,
            contrast_factor=contrast_val,
            clean_background=clean_bg
        )

        st.write("##### Preprocessing Pipeline Inspection:")
        p_col1, p_col2 = st.columns(2)
        with p_col1:
            st.image(active_image, caption="Original Input Image", use_container_width=True)
        with p_col2:
            st.image(
                preview_28x28.resize((200, 200), Image.Resampling.NEAREST),
                caption=f"CNN Input Tensor (28x28 Grayscale, Inverted: {was_inverted})",
                use_container_width=True
            )

        if was_inverted:
            st.caption("ℹ️ *Light background detected: automatically inverted to dark to match Fashion-MNIST.*")
    else:
        st.info("Upload a clothing photo above or select a sample from the left sidebar to begin.")


with col_right:
    st.markdown('<div class="step-header"><span class="step-badge">STEP 2</span> CNN Model Prediction</div>', unsafe_allow_html=True)

    if active_image is not None and model is not None:
        with st.spinner("Analyzing image features with CNN..."):
            predictions = model.predict(input_tensor, verbose=0)[0]

        top_class_idx = int(np.argmax(predictions))
        top_class_name = CLASS_NAMES[top_class_idx]
        confidence_score = float(predictions[top_class_idx]) * 100.0

        # Top Prediction Card
        st.markdown(f"""
        <div class="metric-card">
            <div class="card-label">Predicted Category</div>
            <div class="prediction-badge">
                {top_class_name}
            </div>
            <div class="confidence-text">
                Confidence Score: <span class="confidence-val">{confidence_score:.2f}%</span>
            </div>
            <div class="desc-text">
                {CLASS_DESCRIPTIONS.get(top_class_name, "")}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.progress(min(max(confidence_score / 100.0, 0.0), 1.0))

        # Top 3 Predictions
        st.write("##### Top 3 Candidate Predictions:")
        top3_indices = np.argsort(predictions)[-3:][::-1]
        for rank, idx in enumerate(top3_indices, 1):
            c_name = CLASS_NAMES[idx]
            c_score = predictions[idx] * 100.0
            st.write(f"**#{rank}** **{c_name}**: `{c_score:.2f}%`")

        # Explain Upper-Body Categories Distinction
        st.markdown("""
        <div class="tip-callout">
            <strong>Key Upper-Body Structural Differences:</strong><br>
            • <strong>T-shirt/top:</strong> Short sleeves, simple round/V-neck collar, no front buttons.<br>
            • <strong>Shirt:</strong> Button-down front line with a formal structured collar.<br>
            • <strong>Coat:</strong> Long outerwear, wide lapels, thicker silhouette extending lower.<br>
            • <strong>Pullover:</strong> Long sleeves, closed front knitwear or sweatshirt.
        </div>
        """, unsafe_allow_html=True)

        # Full Bar Chart
        st.write("##### Probability Distribution Across 10 Classes:")
        chart_data = {
            "Fashion Category": CLASS_NAMES,
            "Probability (%)": [float(p * 100.0) for p in predictions]
        }
        st.bar_chart(
            data=chart_data,
            x="Fashion Category",
            y="Probability (%)",
            color="#2563EB"
        )
    else:
        st.markdown("""
        <div class="empty-state-box">
            <div class="empty-state-title">Awaiting Image Input</div>
            <p>Upload a fashion item photograph or select a benchmark sample to trigger real-time CNN feature extraction and classification.</p>
        </div>
        """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 7. Bottom Section: Technical Details for College Presentations
# -----------------------------------------------------------------------------
st.write("---")
with st.expander("Technical Architecture & Viva Reference", expanded=False):
    tab1, tab2, tab3 = st.tabs(["CNN Architecture", "Class Differentiation", "Letterboxing Preprocessing"])

    with tab1:
        st.markdown("""
        ### Enhanced VGG-Style CNN Architecture:
        - **Data Augmentation:** Horizontal flips, translations (±4%), and zooms (±4%) applied during training.
        - **Dual Conv Blocks:** Two consecutive 3×3 convolutions per block (32 → 64 → 128) capturing fine textures and structural features.
        - **Batch Normalization:** Stabilizes gradient propagation and speeds up convergence.
        - **Dropout Regularization:** 20% to 40% dropout prevents feature co-adaptation and overfitting.
        - **Dense & Softmax:** Outputs calibrated class probabilities summing to 1.0 (100%).
        """)

    with tab2:
        st.markdown("""
        ### Distinguishing Shirts, T-shirts, and Coats:
        - Both garments share general torso silhouettes.
        - The primary distinguishing cues are **high-frequency structural details** (collars and front button rows).
        - At 28×28 resolution, preserving aspect ratio prevents a slim shirt from expanding laterally into an overcoat.
        """)

    with tab3:
        st.markdown("""
        ### Aspect-Ratio Preservation (Letterboxing):
        - Direct square resizing squashes rectangular 3:4 smartphone photos, making garments 33% wider than normal.
        - Letterboxing scales the largest dimension to 24 pixels and pads the remaining space with a pure black background (0).
        - This preserves natural proportions and aligns user uploads with the Fashion-MNIST training domain.
        """)
