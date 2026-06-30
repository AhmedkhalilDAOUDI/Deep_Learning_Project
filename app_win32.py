import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os
import io
import urllib.request
import warnings

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

st.set_page_config(page_title="Dogs vs Cats Classifier", page_icon="🐾")

MODEL_PATH = r"C:\Users\Khalil\Desktop\Deep Learning project\data_cats_and_dogs\saved_model"
IMG_SIZE = (160, 160)

with st.spinner("Loading model..."):
    model = load_model(MODEL_PATH)
st.success("Model loaded!")
test_arr = np.zeros((1, IMG_SIZE[0], IMG_SIZE[1], 3), dtype="float32")
with st.spinner("Test run..."):
    prob = float(model.predict(test_arr, verbose=0)[0][0])

label = "dog" if prob > 0.5 else "cat"
st.write(f"Test classification: {label}  (raw prob: {prob:.4f})")
