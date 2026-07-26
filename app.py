import numpy as np
import streamlit as st
import tensorflow as tf
from huggingface_hub import hf_hub_download
from PIL import Image

# Title & Group Information
st.title("Bee vs Fly Classifier 🐝🪰")
st.subheader("CE11 (Mini Project)")
st.write(
    "Upload an image and the trained deep learning model will classify it."
)


# Download & Cache Model from Hugging Face
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="fortuneuwemedimoh01/bee_vs_fly_model",
        filename="bee_vs_fly_model.keras",
    )
    return tf.keras.models.load_model(model_path)


model = load_model()

# Image Uploader
uploaded_file = st.file_uploader(
    "Choose an image...", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocessing
    img_resized = image.resize((150, 150))
    img_array = np.array(img_resized) / 255.0  # Normalized to [0, 1]
    img_batch = np.expand_dims(img_array, axis=0)

    # Prediction
    predictions = model.predict(img_batch)[0]

    if len(predictions) == 1:
        fly_prob = float(predictions[0])
        bee_prob = 1.0 - fly_prob
    else:
        bee_prob = float(predictions[0])
        fly_prob = float(predictions[1])

    st.subheader("Predictions:")

    # Progress bars layout
    st.write(f"Bee: {bee_prob * 100:.2f}%")
    st.progress(bee_prob)

    st.write(f"Fly: {fly_prob * 100:.2f}%")
    st.progress(fly_prob)
    
