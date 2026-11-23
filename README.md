# Fashion Product Classification using CNN in Python

A beginner-friendly, end-to-end Deep Learning project for classifying fashion product images into 10 distinct categories using a Convolutional Neural Network (CNN) built with **TensorFlow / Keras** and deployed with an interactive **Streamlit** web application.

---

## 📌 Project Overview

This project implements image classification on the **Fashion-MNIST** benchmark dataset (developed by Zalando Research). Rather than traditional machine learning, this project utilizes a **Convolutional Neural Network (CNN)**, which automatically learns spatial hierarchies of features (edges, textures, patterns, and complex object shapes).

The complete workflow includes:
1. Loading the 70,000-image Fashion-MNIST dataset directly through Keras.
2. Data preprocessing: pixel normalization (`0-255` -> `0.0-1.0`) and tensor reshaping to `(28, 28, 1)`.
3. Constructing a deep CNN with `Conv2D`, `BatchNormalization`, `MaxPooling2D`, `Dropout`, and `Dense` layers.
4. Model training and evaluation on an unseen test dataset of 10,000 images, achieving **~91-93% test accuracy**.
5. Saving the trained model (`.keras` and `.h5`), training curves plot, and metadata JSON in `models/`.
6. Deploying a clean, modern **Streamlit web application** (`app.py`) allowing users to upload clothing images or select preset Fashion-MNIST test samples, view real-time predictions, inspect confidence scores, and view the full probability distribution.

---

## 🏷️ The 10 Fashion Classes Explained

| Class ID | Category Name | Description |
|:---:|:---|:---|
| **0** | **T-shirt/top** | Casual tops, t-shirts, and short-sleeved shirts |
| **1** | **Trouser** | Jeans, pants, slacks, and formal trousers |
| **2** | **Pullover** | Sweaters, long-sleeve knitwear, and sweatshirts |
| **3** | **Dress** | Women's one-piece dresses and gowns |
| **4** | **Coat** | Winter jackets, overcoats, and long outerwear |
| **5** | **Sandal** | Open-toe sandals, flat footwear, and summer flip-flops |
| **6** | **Shirt** | Button-down collared dress shirts and casual shirts |
| **7** | **Sneaker** | Athletic sports shoes, tennis trainers, and sneakers |
| **8** | **Bag** | Handbags, tote bags, backpacks, and purses |
| **9** | **Ankle boot** | High-top leather boots, winter boots, and ankle boots |

---

## 📁 Project Directory Structure

```text
fashion_product_classification/
│
├── models/                         # Saved trained models and metadata
│   ├── fashion_cnn_model.keras     # Trained CNN model (Native Keras 3 format)
│   ├── fashion_cnn_model.h5        # Trained CNN model (HDF5 format)
│   ├── model_metadata.json         # Class names, descriptions, and accuracy
│   └── training_history.png        # Training vs validation accuracy and loss plot
│
├── sample_images/                  # Real test samples for quick app testing
│   ├── sample_t-shirt_top.png
│   ├── sample_trouser.png
│   ├── sample_pullover.png
│   ├── sample_dress.png
│   ├── sample_coat.png
│   ├── sample_sandal.png
│   ├── sample_shirt.png
│   ├── sample_sneaker.png
│   ├── sample_bag.png
│   └── sample_ankle_boot.png
│
├── train_model.py                  # Script to load data, build CNN, train, and save model
├── app.py                          # Streamlit interactive web interface
├── requirements.txt                # List of Python dependencies
└── README.md                       # Comprehensive documentation & college viva guide
```

---

## 🛠️ Installation & Setup (Step-by-Step)

### Step 1: Open Terminal / PowerShell
Open your terminal or PowerShell and navigate to the project folder:
```powershell
cd C:\Users\Asus\.gemini\antigravity\scratch\fashion_product_classification
```

### Step 2: Install Required Libraries
Run the following command to install TensorFlow, Streamlit, Matplotlib, NumPy, and Pillow:
```powershell
pip install -r requirements.txt
```

---

## 🚀 How to Run the Project

### Phase 1: Train the CNN Model
Execute the training script in your terminal:
```powershell
python train_model.py
```

**What this command does:**
1. Downloads the Fashion-MNIST dataset directly from Keras (cached locally).
2. Normalizes pixel values to `[0.0, 1.0]` and reshapes to `(28, 28, 1)`.
3. Builds the CNN with 3 Convolution blocks and a dense classification head.
4. Trains the model for 10 epochs using a 90% train / 10% validation split.
5. Evaluates the model on the 10,000 test images and prints test accuracy (`~91.5%`).
6. Saves `fashion_cnn_model.keras`, `model_metadata.json`, and `training_history.png` in `models/`.
7. Exports 10 sample images into `sample_images/` for instant testing.

**Expected Terminal Output:**
```text
==================================================================
    Fashion Product Classification - CNN Model Training Pipeline
==================================================================
...
Epoch 10/10 - loss: 0.1784 - accuracy: 0.9324 - val_loss: 0.2241 - val_accuracy: 0.9180
...
========================================================
  >>> FINAL TEST ACCURACY : 91.85% <<<
  >>> FINAL TEST LOSS     : 0.2310 <<<
========================================================
-> Model saved in native format: models/fashion_cnn_model.keras
-> Metadata and class labels saved to: models/model_metadata.json
-> Training history plot saved to: models/training_history.png
-> 10 sample test images saved to: sample_images/
```

---

### Phase 2: Launch the Streamlit Web Application
Once the model is trained, start the interactive web app:
```powershell
streamlit run app.py
```

**Streamlit will launch the web server and print:**
```text
  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

Open Google Chrome and visit:
👉 **`http://localhost:8501`**

---

## 🧠 Convolutional Neural Network (CNN) Architecture Explained

```text
Input Image (28x28x1 Grayscale)
       │
       ▼
[ Conv2D (32 filters, 3x3) + ReLU ]   ──> Extracts low-level features (edges, corners)
[ Batch Normalization ]               ──> Stabilizes distribution of activations
[ MaxPooling2D (2x2) ]                ──> Spatial reduction to 14x14
[ Dropout (0.25) ]                    ──> Randomly drops 25% neurons to prevent overfitting
       │
       ▼
[ Conv2D (64 filters, 3x3) + ReLU ]   ──> Extracts mid-level features (patterns, shapes)
[ Batch Normalization ]
[ MaxPooling2D (2x2) ]                ──> Spatial reduction to 7x7
[ Dropout (0.25) ]
       │
       ▼
[ Conv2D (128 filters, 3x3) + ReLU ]  ──> Extracts high-level class-specific structural features
[ Batch Normalization ]
[ Dropout (0.30) ]
       │
       ▼
[ Flatten Layer ]                     ──> Flattens 2D feature maps to a 1D vector (6272 features)
       │
       ▼
[ Dense Layer (128 units) + ReLU ]    ──> Learns non-linear combinations of feature representations
[ Batch Normalization ]
[ Dropout (0.40) ]                    ──> Regularization before classification
       │
       ▼
[ Dense Layer (10 units) + Softmax ]  ──> Probability distribution across 10 fashion classes
```

---

## 💡 Important Note on Fashion-MNIST vs Real-World Photos

- **Fashion-MNIST Characteristic:** Images are **28×28 grayscale thumbnails** with **white/bright items centered on a solid black background**.
- **Real-World Photos:** Photos taken from smartphones or online shopping sites typically have **white or colorful backgrounds** and high resolutions.
- **Why Invert Background?** If you upload a photo with a white background, the CNN will perceive the white background as the garment. In the Streamlit app, make sure to check the **"Invert Colors"** box so the background turns dark and the object turns bright, matching the training domain!

---

## 🎓 College Viva / Project Defense Q&A

**Q1: Why use a CNN instead of a standard Artificial Neural Network (ANN / MLP)?**
> *Answer:* An ANN flattens the image into 784 1D values immediately, destroying 2D spatial relationships between neighboring pixels. A CNN preserves spatial structure through 2D convolutions, requires far fewer parameters through weight sharing, and is translation invariant.

**Q2: What is the purpose of MaxPooling?**
> *Answer:* MaxPooling reduces the spatial dimensions (height and width) of feature maps by picking the maximum value in local windows. This decreases computational complexity, prevents overfitting, and introduces slight invariance to translation/shifts.

**Q3: Why normalize pixel values by dividing by 255.0?**
> *Answer:* Raw pixel values range from 0 to 255. Unscaled inputs can cause large gradients, leading to numerical overflow, slow convergence, or getting stuck in local minima. Scaling them to `[0.0, 1.0]` provides smooth, stable gradient descent.

**Q4: What activation functions are used and why?**
> *Answer:* 
> - **ReLU (Rectified Linear Unit):** Used in hidden layers to introduce non-linearity and prevent vanishing gradients.
> - **Softmax:** Used in the final output layer to turn raw logits into a normalized probability distribution where all 10 probabilities sum to 1.0 (100%).

**Q5: What is Dropout?**
> *Answer:* Dropout is a regularization technique where a randomly selected fraction of neurons are ignored during forward and backward passes of each training step. This forces the network to learn robust, redundant representations rather than memorizing individual training samples.

---

## 🔧 Troubleshooting

- **Error: `ModuleNotFoundError: No module named 'tensorflow'`**
  - Run `pip install tensorflow streamlit matplotlib pillow` in your terminal.
- **Warning: `Trained Model Not Found inside models/ folder!`**
  - Make sure you ran `python train_model.py` first. Verify that `models/fashion_cnn_model.keras` exists.
- **Port 8501 is already in use:**
  - Launch Streamlit on a custom port: `streamlit run app.py --server.port 8502`.
