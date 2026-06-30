import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import os
import io
import urllib.request
import warnings

# Suppress logs
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tf.get_logger().setLevel('ERROR')

st.set_page_config(page_title="Dogs vs Cats Classifier", page_icon="🐾")

MODEL_DIR = os.path.join(os.path.dirname(__file__), "data_cats_and_dogs", "deploy_model")
IMG_SIZE = (160, 160)

@st.cache_resource
def load_model_cached():
    return tf.saved_model.load(MODEL_DIR)

with st.spinner("Loading model..."):
    model = load_model_cached()
    infer = model.signatures['serving_default']
    st.success("Model loaded!")

st.title("🐕 Dogs vs Cats Classifier 🐈")
st.write("Upload an image **or paste a URL** from the web to classify it.")

# Two input modes: upload + URL
col_upload, col_url = st.columns(2)

with col_upload:
    uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

with col_url:
    url_input = st.text_input("Or paste an image URL:", placeholder="https://example.com/cat.jpg")
    url_btn = st.button("Load from URL")

# Resolve image from either source
image = None
source = None

if url_btn and url_input.strip():
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        req = urllib.request.Request(url_input.strip(), headers=headers)
        with urllib.request.urlopen(req, timeout=15) as resp:
            image = Image.open(io.BytesIO(resp.read())).convert("RGB")
        source = "url"
        st.image(image, caption=f"Loaded from: {url_input[:80]}", use_column_width=True)
    except Exception as e:
        st.error(f"Failed to load image: {e}")

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    source = "upload"
    st.image(image, caption="Uploaded Image", use_column_width=True)

if image is not None:
    # Preprocess
    img_resized = ImageOps.fit(image, IMG_SIZE, Image.LANCZOS)
    img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0).astype(np.float32)

    # Predict
    with st.spinner("Classifying..."):
        out = infer(tf.convert_to_tensor(img_array, dtype=tf.float32))
        prob = float(out['dense_5'].numpy()[0][0])

    label = "dog" if prob > 0.5 else "cat"
    confidence = prob if prob > 0.5 else 1.0 - prob

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Prediction", label.capitalize())
    with col2:
        st.metric("Confidence", f"{confidence:.1%}")

    # Progress bar
    bar_col1, bar_col2 = st.columns([3, 1])
    with bar_col1:
        st.progress(confidence)
    with bar_col2:
        st.write("")
    st.write(f"This is a **{'🐕 DOG' if label == 'dog' else '🐈 CAT'}**  ({confidence:.1%} confident)")

    # Confidence scale
    st.caption(f"Raw logit: {prob:.4f}  |  Threshold: 0.5  →  {'Dog' if prob > 0.5 else 'Cat'}")
