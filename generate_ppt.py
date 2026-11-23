"""
=============================================================================
Generate PowerPoint Presentation for Fashion Product Classification Project
=============================================================================
Features:
- Slide 1: Official Malnad College of Engineering Cover Slide with Logo,
           Project Title, Subject: DEEP LEARNING, Submitted by (Left) & Submitted to (Right)
- Slides 2-7: Comprehensive technical project content
- Slide 8: ONLY "Thank You"
=============================================================================
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PPTX = os.path.join(SCRIPT_DIR, "Fashion_Product_Classification_Presentation.pptx")
LOGO_IMG = os.path.join(SCRIPT_DIR, "college_logo.jpg")
CHART_IMG = os.path.join(SCRIPT_DIR, "models", "training_history.png")

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
blank_slide_layout = prs.slide_layouts[6]

COLOR_BG_DARK = RGBColor(15, 23, 42)       # #0f172a
COLOR_BG_LIGHT = RGBColor(248, 250, 252)   # #f8fafc
COLOR_CARD = RGBColor(255, 255, 255)       # Pure White
COLOR_CARD_BORDER = RGBColor(226, 232, 240)# #e2e8f0
COLOR_PRIMARY = RGBColor(37, 99, 235)      # #2563eb
COLOR_ACCENT = RGBColor(14, 165, 233)      # #0ea5e9
COLOR_TEXT_MAIN = RGBColor(15, 23, 42)     # #0f172a
COLOR_TEXT_MUTED = RGBColor(100, 116, 139) # #64748b
COLOR_WHITE = RGBColor(255, 255, 255)
COLOR_GREEN = RGBColor(16, 185, 129)      # #10b981
COLOR_AMBER = RGBColor(245, 158, 11)      # #f59e0b


def set_slide_background(slide, color):
    bg_shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    bg_shape.fill.solid()
    bg_shape.fill.fore_color.rgb = color
    bg_shape.line.fill.background()
    return bg_shape


def add_header(slide, title_text, category_text="DEEP LEARNING PROJECT PRESENTATION"):
    cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.35))
    tf_cat = cat_box.text_frame
    tf_cat.word_wrap = True
    p_cat = tf_cat.paragraphs[0]
    p_cat.text = category_text.upper()
    p_cat.font.name = "Calibri"
    p_cat.font.size = Pt(11)
    p_cat.font.bold = True
    p_cat.font.color.rgb = COLOR_PRIMARY

    title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.7), Inches(11.7), Inches(0.7))
    tf_title = title_box.text_frame
    tf_title.word_wrap = True
    p_title = tf_title.paragraphs[0]
    p_title.text = title_text
    p_title.font.name = "Calibri"
    p_title.font.size = Pt(24)
    p_title.font.bold = True
    p_title.font.color.rgb = COLOR_TEXT_MAIN


def add_card(slide, left, top, width, height, title, items, border_color=COLOR_PRIMARY):
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    card.fill.solid()
    card.fill.fore_color.rgb = COLOR_CARD
    card.line.color.rgb = COLOR_CARD_BORDER
    card.line.width = Pt(1.5)

    strip = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left + Inches(0.02), top + Inches(0.02), Inches(0.12), height - Inches(0.04))
    strip.fill.solid()
    strip.fill.fore_color.rgb = border_color
    strip.line.fill.background()

    tb = slide.shapes.add_textbox(left + Inches(0.35), top + Inches(0.2), width - Inches(0.5), height - Inches(0.4))
    tf = tb.text_frame
    tf.word_wrap = True

    p_t = tf.paragraphs[0]
    p_t.text = title
    p_t.font.name = "Calibri"
    p_t.font.size = Pt(18)
    p_t.font.bold = True
    p_t.font.color.rgb = COLOR_TEXT_MAIN
    p_t.space_after = Pt(12)

    for item in items:
        p = tf.add_paragraph()
        p.text = item
        p.font.name = "Calibri"
        p.font.size = Pt(13)
        p.font.color.rgb = COLOR_TEXT_MUTED
        p.space_after = Pt(8)


# =============================================================================
# SLIDE 1: OFFICIAL MALNAD COLLEGE COVER SLIDE (WITH LOGO & DETAILS)
# =============================================================================
slide1 = prs.slides.add_slide(blank_slide_layout)
set_slide_background(slide1, RGBColor(255, 255, 255))

# Top Border Accent
top_accent = slide1.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(0.12))
top_accent.fill.solid()
top_accent.fill.fore_color.rgb = COLOR_PRIMARY
top_accent.line.fill.background()

# College Logo (Top Middle)
if os.path.exists(LOGO_IMG):
    # Centered: slide width is 13.333, logo width is 1.6, left = (13.333 - 1.6)/2 = 5.866
    logo_left = Inches(5.866)
    logo_top = Inches(0.4)
    slide1.shapes.add_picture(LOGO_IMG, logo_left, logo_top, width=Inches(1.6), height=Inches(1.8))

# College Name (Followed by logo)
tb_college = slide1.shapes.add_textbox(Inches(1.0), Inches(2.35), Inches(11.333), Inches(0.6))
tf_college = tb_college.text_frame
tf_college.word_wrap = True
p_c = tf_college.paragraphs[0]
p_c.text = "MALNAD COLLEGE OF ENGINEERING"
p_c.alignment = PP_ALIGN.CENTER
p_c.font.name = "Calibri"
p_c.font.size = Pt(26)
p_c.font.bold = True
p_c.font.color.rgb = RGBColor(15, 23, 42)

p_dept = tf_college.add_paragraph()
p_dept.text = "DEPARTMENT OF INFORMATION SCIENCE AND ENGINEERING"
p_dept.alignment = PP_ALIGN.CENTER
p_dept.font.name = "Calibri"
p_dept.font.size = Pt(13)
p_dept.font.bold = True
p_dept.font.color.rgb = COLOR_PRIMARY
p_dept.space_before = Pt(4)

# Project Title
tb_proj = slide1.shapes.add_textbox(Inches(1.0), Inches(3.3), Inches(11.333), Inches(0.9))
tf_proj = tb_proj.text_frame
tf_proj.word_wrap = True
p_proj = tf_proj.paragraphs[0]
p_proj.text = "Fashion Product Classification using CNN in Python"
p_proj.alignment = PP_ALIGN.CENTER
p_proj.font.name = "Calibri"
p_proj.font.size = Pt(28)
p_proj.font.bold = True
p_proj.font.color.rgb = RGBColor(29, 78, 216)

# Subject Pill
tb_subj = slide1.shapes.add_textbox(Inches(1.0), Inches(4.3), Inches(11.333), Inches(0.5))
tf_subj = tb_subj.text_frame
tf_subj.word_wrap = True
p_subj = tf_subj.paragraphs[0]
p_subj.text = "SUBJECT : DEEP LEARNING"
p_subj.alignment = PP_ALIGN.CENTER
p_subj.font.name = "Calibri"
p_subj.font.size = Pt(15)
p_subj.font.bold = True
p_subj.font.color.rgb = RGBColor(71, 85, 105)

# Separator Line
line = slide1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(1.5), Inches(4.95), Inches(10.333), Inches(0.02))
line.fill.solid()
line.fill.fore_color.rgb = RGBColor(226, 232, 240)
line.line.fill.background()

# --- Left Bottom Corner: Submitted by ---
card_sub_by = slide1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.2), Inches(5.2), Inches(4.8), Inches(1.8))
card_sub_by.fill.solid()
card_sub_by.fill.fore_color.rgb = RGBColor(248, 250, 252)
card_sub_by.line.color.rgb = RGBColor(226, 232, 240)
tf_sb = card_sub_by.text_frame
tf_sb.word_wrap = True

p_sb0 = tf_sb.paragraphs[0]
p_sb0.text = "Submitted by,"
p_sb0.font.name = "Calibri"
p_sb0.font.size = Pt(13)
p_sb0.font.bold = True
p_sb0.font.color.rgb = COLOR_PRIMARY

p_sb1 = tf_sb.add_paragraph()
p_sb1.text = "A Y KAUSTUBHA SAI"
p_sb1.font.name = "Calibri"
p_sb1.font.size = Pt(17)
p_sb1.font.bold = True
p_sb1.font.color.rgb = RGBColor(15, 23, 42)
p_sb1.space_before = Pt(4)

p_sb2 = tf_sb.add_paragraph()
p_sb2.text = "USN: 4MC23IS002"
p_sb2.font.name = "Calibri"
p_sb2.font.size = Pt(13)
p_sb2.font.bold = True
p_sb2.font.color.rgb = RGBColor(51, 65, 85)

p_sb3 = tf_sb.add_paragraph()
p_sb3.text = "[ISE DEPT]"
p_sb3.font.name = "Calibri"
p_sb3.font.size = Pt(12)
p_sb3.font.bold = True
p_sb3.font.color.rgb = RGBColor(100, 116, 139)

# --- Right Bottom Corner: Submitted to ---
card_sub_to = slide1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.333), Inches(5.2), Inches(4.8), Inches(1.8))
card_sub_to.fill.solid()
card_sub_to.fill.fore_color.rgb = RGBColor(248, 250, 252)
card_sub_to.line.color.rgb = RGBColor(226, 232, 240)
tf_st = card_sub_to.text_frame
tf_st.word_wrap = True

p_st0 = tf_st.paragraphs[0]
p_st0.text = "Submitted to,"
p_st0.font.name = "Calibri"
p_st0.font.size = Pt(13)
p_st0.font.bold = True
p_st0.font.color.rgb = COLOR_GREEN

p_st1 = tf_st.add_paragraph()
p_st1.text = "Mrs. Vanitha A"
p_st1.font.name = "Calibri"
p_st1.font.size = Pt(17)
p_st1.font.bold = True
p_st1.font.color.rgb = RGBColor(15, 23, 42)
p_st1.space_before = Pt(4)

p_st2 = tf_st.add_paragraph()
p_st2.text = "Assistant Professor"
p_st2.font.name = "Calibri"
p_st2.font.size = Pt(13)
p_st2.font.color.rgb = RGBColor(51, 65, 85)

p_st3 = tf_st.add_paragraph()
p_st3.text = "[ISE DEPT]"
p_st3.font.name = "Calibri"
p_st3.font.size = Pt(12)
p_st3.font.bold = True
p_st3.font.color.rgb = RGBColor(100, 116, 139)


# =============================================================================
# SLIDE 2: Project Overview & Tech Stack (Dark Theme)
# =============================================================================
slide2 = prs.slides.add_slide(blank_slide_layout)
set_slide_background(slide2, COLOR_BG_DARK)

badge = slide2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), Inches(1.5), Inches(3.6), Inches(0.45))
badge.fill.solid()
badge.fill.fore_color.rgb = RGBColor(30, 41, 59)
badge.line.color.rgb = COLOR_ACCENT
tf_b = badge.text_frame
p_b = tf_b.paragraphs[0]
p_b.text = "PROJECT ARCHITECTURE & STACK"
p_b.alignment = PP_ALIGN.CENTER
p_b.font.size = Pt(11)
p_b.font.bold = True
p_b.font.color.rgb = COLOR_ACCENT

tb_t2 = slide2.shapes.add_textbox(Inches(1.0), Inches(2.2), Inches(11.3), Inches(1.8))
tf_t2 = tb_t2.text_frame
tf_t2.word_wrap = True
p_t2 = tf_t2.paragraphs[0]
p_t2.text = "Fashion Product Classification"
p_t2.font.name = "Calibri"
p_t2.font.size = Pt(40)
p_t2.font.bold = True
p_t2.font.color.rgb = COLOR_WHITE

p_t2_sub = tf_t2.add_paragraph()
p_t2_sub.text = "Using Deep Convolutional Neural Networks (CNN) in Python"
p_t2_sub.font.name = "Calibri"
p_t2_sub.font.size = Pt(26)
p_t2_sub.font.bold = True
p_t2_sub.font.color.rgb = COLOR_ACCENT
p_t2_sub.space_before = Pt(8)

tb_desc = slide2.shapes.add_textbox(Inches(1.0), Inches(4.3), Inches(11.3), Inches(1.0))
tf_desc = tb_desc.text_frame
tf_desc.word_wrap = True
p_desc = tf_desc.paragraphs[0]
p_desc.text = "An end-to-end computer vision system trained on the Fashion-MNIST dataset, featuring automated image preprocessing, high-accuracy neural inference, and an interactive Streamlit web deployment."
p_desc.font.name = "Calibri"
p_desc.font.size = Pt(15)
p_desc.font.color.rgb = RGBColor(203, 213, 225)

meta_y = Inches(5.6)
meta_h = Inches(1.1)

m1 = slide2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), meta_y, Inches(5.4), meta_h)
m1.fill.solid()
m1.fill.fore_color.rgb = RGBColor(30, 41, 59)
m1.line.color.rgb = RGBColor(51, 65, 85)
tf_m1 = m1.text_frame
tf_m1.word_wrap = True
p_m1 = tf_m1.paragraphs[0]
p_m1.text = "Technologies Used:"
p_m1.font.size = Pt(12)
p_m1.font.bold = True
p_m1.font.color.rgb = COLOR_ACCENT
p_m1_b = tf_m1.add_paragraph()
p_m1_b.text = "Python • TensorFlow / Keras • Streamlit • NumPy • Pillow"
p_m1_b.font.size = Pt(13)
p_m1_b.font.color.rgb = COLOR_WHITE

m2 = slide2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), meta_y, Inches(5.5), meta_h)
m2.fill.solid()
m2.fill.fore_color.rgb = RGBColor(30, 41, 59)
m2.line.color.rgb = RGBColor(51, 65, 85)
tf_m2 = m2.text_frame
tf_m2.word_wrap = True
p_m2 = tf_m2.paragraphs[0]
p_m2.text = "Project Status & Demo:"
p_m2.font.size = Pt(12)
p_m2.font.bold = True
p_m2.font.color.rgb = COLOR_GREEN
p_m2_b = tf_m2.add_paragraph()
p_m2_b.text = "Trained & Evaluated (92.74% Accuracy) • Live at http://localhost:8501"
p_m2_b.font.size = Pt(13)
p_m2_b.font.color.rgb = COLOR_WHITE


# =============================================================================
# SLIDE 3: Problem Statement & Objectives
# =============================================================================
slide3 = prs.slides.add_slide(blank_slide_layout)
set_slide_background(slide3, COLOR_BG_LIGHT)
add_header(slide3, "Project Motivation, Problem Statement & Objectives")

add_card(
    slide3, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.2),
    "1. The Real-World Challenge",
    [
        "• High-Volume E-Commerce: Platforms like Amazon and Myntra process millions of new product images daily from third-party sellers.",
        "• Manual Cataloging Bottleneck: Manually inspecting, tagging, and filing each product photo into categories is labor-intensive, slow, and expensive.",
        "• Human Error & Ambiguity: Subtle design overlaps (e.g., T-shirt vs Shirt vs Coat) cause inconsistent tagging and poor customer search results.",
        "• Need for Automation: E-commerce demands millisecond-level, automated visual recognition to streamline catalogs and enable image-based shopping."
    ],
    border_color=COLOR_AMBER
)

add_card(
    slide3, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.2),
    "2. Key Project Objectives",
    [
        "• Robust CNN Architecture: Design and train a deep Convolutional Neural Network capable of classifying 10 distinct fashion categories.",
        "• High Test Accuracy: Surpass 92% evaluation accuracy on 10,000 completely unseen benchmark test images.",
        "• Real-World Image Preprocessing: Solve domain shift issues (e.g. aspect ratio squashing and background differences) when handling custom phone photos.",
        "• Interactive Web Deployment: Deploy a responsive Streamlit web application providing instant predictions and probability breakdowns."
    ],
    border_color=COLOR_PRIMARY
)


# =============================================================================
# SLIDE 4: Dataset Overview (Fashion-MNIST)
# =============================================================================
slide4 = prs.slides.add_slide(blank_slide_layout)
set_slide_background(slide4, COLOR_BG_LIGHT)
add_header(slide4, "Dataset Overview: Zalando's Fashion-MNIST Benchmark")

add_card(
    slide4, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.2),
    "Dataset Architecture & Scale",
    [
        "• Benchmark Origin: Developed by Zalando Research as a more challenging modern drop-in replacement for classic MNIST handwritten digits.",
        "• 70,000 Total Images: Partitioned into 60,000 training examples and 10,000 test examples.",
        "• Balanced Distribution: Exactly 7,000 images per class across 10 distinct categories, preventing class-imbalance bias.",
        "• Image Properties: 28 x 28 pixels, single-channel grayscale images with normalized centered objects.",
        "• Real-World Complexity: Contains intricate textures, silhouettes, collar styles, and overlapping visual features."
    ],
    border_color=COLOR_PRIMARY
)

add_card(
    slide4, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.2),
    "The 10 Fashion Classes & Preprocessing",
    [
        "• 0: T-shirt/top  | Short-sleeved casual crewnecks / tops",
        "• 1: Trouser      | Jeans, slacks, and formal trousers",
        "• 2: Pullover     | Sweaters, sweatshirts, and knitwear",
        "• 3: Dress        | Women's one-piece dresses and gowns",
        "• 4: Coat         | Long winter outerwear and overcoats",
        "• 5: Sandal       | Open-toe footwear and summer flip-flops",
        "• 6: Shirt        | Button-down dress shirts with collars",
        "• 7: Sneaker      | Athletic sports trainers and shoes",
        "• 8: Bag          | Handbags, backpacks, and luggage",
        "• 9: Ankle boot   | High-top winter and leather boots",
        "• Normalization: Scaled raw pixels from [0, 255] to [0.0, 1.0]."
    ],
    border_color=COLOR_GREEN
)


# =============================================================================
# SLIDE 5: Model Architecture & CNN Design
# =============================================================================
slide5 = prs.slides.add_slide(blank_slide_layout)
set_slide_background(slide5, COLOR_BG_LIGHT)
add_header(slide5, "Deep Convolutional Neural Network (CNN) Architecture")

add_card(
    slide5, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.2),
    "Why CNN over Traditional ANN?",
    [
        "• Preserves Spatial 2D Geometry: An ANN flattens 28x28 into 784 numbers, losing spatial pixel adjacency. CNNs preserve 2D topology.",
        "• Weight Sharing: Slides identical 3x3 convolution kernels across the image, drastically cutting parameter count.",
        "• Translation Invariance: Recognizes clothing items even if they are shifted or off-center.",
        "• Data Augmentation Integrated: Random horizontal flips, small translations, and zooms force the model to learn invariant clothing features."
    ],
    border_color=COLOR_PRIMARY
)

add_card(
    slide5, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.2),
    "Enhanced VGG-Style Dual Conv Pipeline",
    [
        "• Conv Block 1: Dual Conv2D(32, 3x3) + BatchNorm + MaxPool(2x2) + Dropout(0.20) -> Captures edge & corner primitives.",
        "• Conv Block 2: Dual Conv2D(64, 3x3) + BatchNorm + MaxPool(2x2) + Dropout(0.25) -> Learns mid-level outlines & shapes.",
        "• Conv Block 3: Dual Conv2D(128, 3x3) + BatchNorm + MaxPool(2x2) + Dropout(0.30) -> Learns high-level category structures.",
        "• Dense Classification Head: Flatten (6,272 units) -> Dense(256) -> BatchNorm -> Dropout(0.40) -> Dense(128) -> Dense(10, Softmax).",
        "• Dynamic Optimizer: Adam with ReduceLROnPlateau learning rate decay."
    ],
    border_color=COLOR_ACCENT
)


# =============================================================================
# SLIDE 6: Real-World Preprocessing & Web Deployment
# =============================================================================
slide6 = prs.slides.add_slide(blank_slide_layout)
set_slide_background(slide6, COLOR_BG_LIGHT)
add_header(slide6, "Solving Real-World Image Challenges & Web Deployment")

add_card(
    slide6, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.2),
    "The Smart Preprocessing Pipeline",
    [
        "• The Challenge: Real smartphone photos have white backgrounds and 3:4 portrait aspect ratios. Direct resizing squashes shirts into bulky coats!",
        "• Aspect-Ratio Preserving Letterboxing: Proportioned thumbnailing (scales longest dimension to 24px) centered onto a 28x28 black canvas.",
        "• Smart Background Auto-Inversion: Evaluates corner pixel intensities. If background is light (>120), inverts it so objects are bright on dark.",
        "• Contrast Stretching & Noise Clamping: Enhances button and collar contrast while zeroing out background shadow haze."
    ],
    border_color=COLOR_AMBER
)

add_card(
    slide6, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.2),
    "Interactive Streamlit Web Interface",
    [
        "• Zero-Latency Local Serving: Built with Streamlit and cached resource loaders (@st.cache_resource) for sub-25ms inference.",
        "• Multi-Format Image Upload: Supports drag-and-drop for JPG, JPEG, and PNG images.",
        "• Quick-Test Benchmark Library: Includes pre-loaded Fashion-MNIST test samples for immediate demonstration without files.",
        "• Transparent Visual Inspection: Side-by-side comparison of original image vs the 28x28 grayscale CNN input tensor.",
        "• Full Probability Distribution: Top-3 candidates with confidence scores and full 10-class interactive bar chart."
    ],
    border_color=COLOR_GREEN
)


# =============================================================================
# SLIDE 7: Results, Performance & Real-World Applications
# =============================================================================
slide7 = prs.slides.add_slide(blank_slide_layout)
set_slide_background(slide7, COLOR_BG_LIGHT)
add_header(slide7, "Experimental Results & Real-World Industry Impact")

add_card(
    slide7, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.2),
    "Key Performance Metrics",
    [
        "• Final Test Accuracy: 92.74% on 10,000 unseen test images.",
        "• Final Test Loss: 0.1969 (Sparse Categorical Cross-Entropy).",
        "• Model Size: 617,450 trainable parameters (~2.36 MB).",
        "• Class Accuracy Highlights:",
        "    - Bags: 99.3% accuracy",
        "    - Trousers: 98.2% accuracy",
        "    - Sandals & Sneakers: 97.5% accuracy",
        "    - Pullovers: 91.2% | Coats: 90.9%",
        "• Solved Tops Confusion: Data augmentation + letterboxing eliminated shirt-to-coat distortion."
    ],
    border_color=COLOR_GREEN
)

add_card(
    slide7, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.2),
    "Practical Industry Applications",
    [
        "• 1. Automated Catalog Tagging: E-commerce platforms auto-classify seller uploads instantly without manual human review.",
        "• 2. Visual Shopping Search: Powers 'Shop the Look' features where users snap a photo of street apparel to find matching items.",
        "• 3. Warehouse Logistics & QA: Overhead cameras on sorting conveyor belts automatically verify garment routing into correct bins.",
        "• 4. Fashion Recommendation Engines: Suggests complementary categories (e.g., matching Sneaker or Bag when viewing a Dress)."
    ],
    border_color=COLOR_PRIMARY
)

if os.path.exists(CHART_IMG):
    try:
        slide7.shapes.add_picture(CHART_IMG, Inches(7.1), Inches(4.7), width=Inches(5.1), height=Inches(1.8))
    except Exception as e:
        pass


# =============================================================================
# SLIDE 8: ONLY "THANK YOU" (Clean, Centered, Elegant)
# =============================================================================
slide8 = prs.slides.add_slide(blank_slide_layout)
set_slide_background(slide8, COLOR_BG_DARK)

tb_thx = slide8.shapes.add_textbox(Inches(1.0), Inches(2.8), Inches(11.333), Inches(2.0))
tf_thx = tb_thx.text_frame
tf_thx.word_wrap = True
p_thx = tf_thx.paragraphs[0]
p_thx.text = "Thank You"
p_thx.alignment = PP_ALIGN.CENTER
p_thx.font.name = "Calibri"
p_thx.font.size = Pt(72)
p_thx.font.bold = True
p_thx.font.color.rgb = COLOR_WHITE

prs.save(OUTPUT_PPTX)
print(f"SUCCESS: Total 8 slides PowerPoint generated at: {OUTPUT_PPTX}")
