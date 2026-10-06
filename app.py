import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import (
    preprocess_input,
    decode_predictions
)
from PIL import Image
import numpy as np
import time


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="VisionAI | Image Recognition",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 10% 10%, rgba(99,102,241,0.18), transparent 30%),
        radial-gradient(circle at 90% 20%, rgba(168,85,247,0.15), transparent 30%),
        radial-gradient(circle at 50% 90%, rgba(14,165,233,0.12), transparent 35%),
        #070b17;
    color: white;
}


/* Remove Streamlit top padding */

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}


/* Header */

.hero {
    text-align: center;
    padding: 35px 20px 25px;
}

.hero-badge {
    display: inline-block;
    padding: 8px 16px;
    border-radius: 50px;
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(129,140,248,0.35);
    color: #a5b4fc;
    font-size: 13px;
    font-weight: 600;
    margin-bottom: 18px;
    animation: glow 2s infinite alternate;
}

.hero-title {
    font-size: 56px;
    font-weight: 800;
    letter-spacing: -2px;
    background: linear-gradient(
        90deg,
        #ffffff,
        #a5b4fc,
        #c084fc,
        #67e8f9
    );
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 10px;
}

.hero-subtitle {
    color: #94a3b8;
    font-size: 18px;
    max-width: 650px;
    margin: auto;
    line-height: 1.7;
}


/* Glass cards */

.glass {
    background: rgba(15,23,42,0.70);
    border: 1px solid rgba(148,163,184,0.14);
    border-radius: 24px;
    padding: 25px;
    backdrop-filter: blur(18px);
    box-shadow: 0 20px 60px rgba(0,0,0,0.25);
}


/* Upload area */

.upload-card {
    border: 2px dashed rgba(129,140,248,0.45);
    border-radius: 22px;
    padding: 35px;
    text-align: center;
    background: rgba(99,102,241,0.05);
    transition: 0.3s;
}

.upload-card:hover {
    border-color: #818cf8;
    background: rgba(99,102,241,0.10);
    transform: translateY(-2px);
}


/* Prediction */

.prediction-card {
    background: linear-gradient(
        135deg,
        rgba(79,70,229,0.25),
        rgba(124,58,237,0.15)
    );
    border: 1px solid rgba(129,140,248,0.35);
    border-radius: 24px;
    padding: 28px;
    text-align: center;
    box-shadow: 0 0 40px rgba(99,102,241,0.12);
}

.prediction-label {
    color: #94a3b8;
    font-size: 14px;
    margin-bottom: 8px;
}

.prediction-name {
    font-size: 36px;
    font-weight: 800;
    color: #ffffff;
}

.confidence {
    color: #a5b4fc;
    font-size: 18px;
    margin-top: 8px;
}


/* Stats */

.stat-card {
    background: rgba(15,23,42,0.65);
    border: 1px solid rgba(148,163,184,0.12);
    border-radius: 18px;
    padding: 20px;
    text-align: center;
}

.stat-icon {
    font-size: 25px;
}

.stat-value {
    font-size: 20px;
    font-weight: 700;
    margin-top: 5px;
}

.stat-label {
    color: #64748b;
    font-size: 12px;
}


/* Section headings */

.section-title {
    font-size: 25px;
    font-weight: 700;
    margin-top: 35px;
    margin-bottom: 18px;
}


/* Feature cards */

.feature {
    background: rgba(15,23,42,0.60);
    border: 1px solid rgba(148,163,184,0.12);
    border-radius: 20px;
    padding: 22px;
    height: 100%;
    transition: all 0.3s ease;
}

.feature:hover {
    transform: translateY(-6px);
    border-color: rgba(129,140,248,0.4);
    box-shadow: 0 15px 35px rgba(0,0,0,0.25);
}

.feature-icon {
    font-size: 32px;
}

.feature-title {
    font-size: 17px;
    font-weight: 700;
    margin-top: 10px;
}

.feature-text {
    color: #94a3b8;
    font-size: 13px;
    line-height: 1.6;
}


/* Sidebar */

section[data-testid="stSidebar"] {
    background: rgba(7,11,23,0.95);
    border-right: 1px solid rgba(148,163,184,0.12);
}


/* Buttons */

.stButton > button {
    width: 100%;
    border-radius: 14px;
    border: 1px solid rgba(129,140,248,0.35);
    background: linear-gradient(
        135deg,
        #4f46e5,
        #7c3aed
    );
    color: white;
    font-weight: 700;
    padding: 12px;
    transition: all 0.25s ease;
}

.stButton > button:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 30px rgba(99,102,241,0.35);
}


/* Progress */

.stProgress > div > div > div > div {
    background: linear-gradient(
        90deg,
        #6366f1,
        #a855f7,
        #22d3ee
    );
}


/* Image */

[data-testid="stImage"] img {
    border-radius: 20px;
    border: 1px solid rgba(148,163,184,0.15);
}


/* Footer */

.footer {
    text-align: center;
    color: #64748b;
    padding: 30px 0 10px;
    font-size: 13px;
}


/* Animation */

@keyframes glow {
    from {
        box-shadow: 0 0 10px rgba(99,102,241,0.15);
    }
    to {
        box-shadow: 0 0 30px rgba(139,92,246,0.35);
    }
}

@keyframes pulse {
    0% { opacity: 0.5; }
    50% { opacity: 1; }
    100% { opacity: 0.5; }
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# MODEL
# =========================================================

@st.cache_resource
def load_model():
    return MobileNetV2(weights="imagenet")


# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="hero">

<div class="hero-badge">
🟢 AI SYSTEM ONLINE • MOBILE NET V2
</div>

<div class="hero-title">
VisionAI
</div>

<div class="hero-subtitle">
An intelligent image recognition system powered by deep learning.
Upload an image and let AI analyze what it sees.
</div>

</div>
""", unsafe_allow_html=True)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## 👁️ VisionAI")

    st.caption("Intelligent Image Recognition")

    st.markdown("---")

    st.markdown("### ⚙️ System")

    st.success("AI Model Online")

    st.markdown("""
    **Model**

    MobileNetV2

    **Framework**

    TensorFlow + Keras

    **Interface**

    Streamlit

    **Input**

    JPG • JPEG • PNG
    """)

    st.markdown("---")

    st.markdown("### 🧠 How it works")

    st.markdown("""
    **01** — Upload image

    **02** — Resize & preprocess

    **03** — Neural network analysis

    **04** — Generate predictions

    **05** — Display confidence
    """)


# =========================================================
# UPLOAD
# =========================================================

st.markdown(
    '<div class="section-title">📤 Upload an Image</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="upload-card">

<h2>🖼️ Drop your image here</h2>

<p style="color:#94a3b8;">
Supported formats: JPG, JPEG and PNG
</p>

</div>
""", unsafe_allow_html=True)


uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)


# =========================================================
# IMAGE PROCESSING
# =========================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.markdown(
        '<div class="section-title">🔬 AI Analysis</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(
        [1, 1],
        gap="large"
    )


    # IMAGE COLUMN

    with col1:

        st.markdown(
            '<div class="glass">',
            unsafe_allow_html=True
        )

        st.markdown("### 📷 Your Image")

        st.image(
            image,
            use_container_width=True
        )

        st.markdown(
            f"""
            <div style="color:#94a3b8;font-size:13px;margin-top:10px;">
            📐 Resolution: {image.width} × {image.height}px
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("</div>", unsafe_allow_html=True)


    # AI COLUMN

    with col2:

        st.markdown(
            '<div class="glass">',
            unsafe_allow_html=True
        )

        st.markdown("### 🤖 AI Prediction")

        with st.spinner("🧠 Neural network analyzing..."):

            time.sleep(0.5)

            resized_image = image.resize(
                (224, 224)
            )

            image_array = np.array(
                resized_image
            )

            image_array = np.expand_dims(
                image_array,
                axis=0
            )

            image_array = preprocess_input(
                image_array
            )

            predictions = model.predict(
                image_array,
                verbose=0
            )

            results = decode_predictions(
                predictions,
                top=5
            )[0]


        top_label = results[0][1]
        top_confidence = results[0][2] * 100


        st.markdown(
            f"""
            <div class="prediction-card">

            <div class="prediction-label">
            🎯 TOP PREDICTION
            </div>

            <div class="prediction-name">
            {top_label}
            </div>

            <div class="confidence">
            {top_confidence:.2f}% confidence
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("</div>", unsafe_allow_html=True)


    # =====================================================
    # STATISTICS
    # =====================================================

    st.markdown(
        '<div class="section-title">📊 Analysis Overview</div>',
        unsafe_allow_html=True
    )

    s1, s2, s3, s4 = st.columns(4)

    with s1:
        st.markdown("""
        <div class="stat-card">
        <div class="stat-icon">🤖</div>
        <div class="stat-value">AI</div>
        <div class="stat-label">MODEL TYPE</div>
        </div>
        """, unsafe_allow_html=True)

    with s2:
        st.markdown("""
        <div class="stat-card">
        <div class="stat-icon">🧠</div>
        <div class="stat-value">1000+</div>
        <div class="stat-label">IMAGE CLASSES</div>
        </div>
        """, unsafe_allow_html=True)

    with s3:
        st.markdown("""
        <div class="stat-card">
        <div class="stat-icon">⚡</div>
        <div class="stat-value">Fast</div>
        <div class="stat-label">INFERENCE</div>
        </div>
        """, unsafe_allow_html=True)

    with s4:
        st.markdown("""
        <div class="stat-card">
        <div class="stat-icon">📈</div>
        <div class="stat-value">Top 5</div>
        <div class="stat-label">PREDICTIONS</div>
        </div>
        """, unsafe_allow_html=True)


    # =====================================================
    # TOP PREDICTIONS
    # =====================================================

    st.markdown(
        '<div class="section-title">🏆 Top 5 Predictions</div>',
        unsafe_allow_html=True
    )

    for rank, (_, label, confidence) in enumerate(
        results,
        start=1
    ):

        percentage = confidence * 100

        st.markdown(
            f"""
            <div style="
            background:rgba(15,23,42,0.65);
            padding:15px 18px;
            border-radius:15px;
            margin-bottom:10px;
            border:1px solid rgba(148,163,184,0.10);
            ">

            <div style="
            display:flex;
            justify-content:space-between;
            margin-bottom:7px;
            ">

            <span>
            <b>#{rank}</b> &nbsp; {label}
            </span>

            <span style="color:#a5b4fc;">
            {percentage:.2f}%
            </span>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.progress(
            float(confidence)
        )

        st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# FEATURES
# =========================================================

else:

    st.markdown(
        '<div class="section-title">✨ What VisionAI Can Do</div>',
        unsafe_allow_html=True
    )

    f1, f2, f3 = st.columns(3)

    with f1:
        st.markdown("""
        <div class="feature">

        <div class="feature-icon">🧠</div>

        <div class="feature-title">
        Deep Learning
        </div>

        <div class="feature-text">
        Powered by MobileNetV2, a pretrained
        convolutional neural network designed
        for efficient image classification.
        </div>

        </div>
        """, unsafe_allow_html=True)


    with f2:
        st.markdown("""
        <div class="feature">

        <div class="feature-icon">⚡</div>

        <div class="feature-title">
        Fast Recognition
        </div>

        <div class="feature-text">
        Analyze an image and receive AI
        predictions within seconds through
        an optimized inference pipeline.
        </div>

        </div>
        """, unsafe_allow_html=True)


    with f3:
        st.markdown("""
        <div class="feature">

        <div class="feature-icon">📊</div>

        <div class="feature-title">
        Confidence Analysis
        </div>

        <div class="feature-text">
        View the top five predictions together
        with their confidence scores for
        better understanding of the result.
        </div>

        </div>
        """, unsafe_allow_html=True)


    # HOW IT WORKS

    st.markdown(
        '<div class="section-title">🔎 How It Works</div>',
        unsafe_allow_html=True
    )

    h1, h2, h3, h4 = st.columns(4)

    steps = [
        ("01", "📤", "Upload", "Choose an image"),
        ("02", "⚙️", "Process", "Prepare the image"),
        ("03", "🧠", "Analyze", "AI studies the image"),
        ("04", "🎯", "Predict", "Generate results")
    ]

    for column, (number, icon, title, text) in zip(
        [h1, h2, h3, h4],
        steps
    ):

        with column:

            st.markdown(
                f"""
                <div class="feature">

                <div style="
                color:#818cf8;
                font-size:12px;
                font-weight:700;
                ">
                STEP {number}
                </div>

                <div class="feature-icon">
                {icon}
                </div>

                <div class="feature-title">
                {title}
                </div>

                <div class="feature-text">
                {text}
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown("""
<div class="footer">

<b style="color:#a5b4fc;">VisionAI</b>

<br>

Intelligent Image Recognition • Built with Python, TensorFlow & Streamlit

<br><br>

© 2026 VisionAI

</div>
""", unsafe_allow_html=True)