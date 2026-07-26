import numpy as np
import streamlit as st
import tensorflow as tf
from huggingface_hub import hf_hub_download
from PIL import Image

# Title & Description
st.title("Bee vs Fly Classifier 🐝🪰")
st.write(
    "Upload an image and the trained deep learning model will classify it."
)

# Class Labels
CLASS_NAMES = ["Bee 🐝", "Fly 🪰"]


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
    "Upload an Image", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocessing (Resize & Normalize)
    img_resized = image.resize((150, 150))
    img_array = np.array(img_resized) / 255.0  # Scale pixel values to [0, 1]
    img_batch = np.expand_dims(img_array, axis=0)

    # Prediction
    predictions = model.predict(img_batch)[0]

    # Binary classification logic
    if len(predictions) == 1:
        score = float(predictions[0])
        # Assuming 0 = Bee, 1 = Fly
        if score > 0.5:
            predicted_class = CLASS_NAMES[1]  # Fly
            confidence = score * 100
        else:
            predicted_class = CLASS_NAMES[0]  # Bee
            confidence = (1.0 - score) * 100
    else:
        predicted_idx = np.argmax(predictions)
        predicted_class = CLASS_NAMES[predicted_idx]
        confidence = float(predictions[predicted_idx]) * 100

    # Group 1 Style Output Display
    st.success(f"**Prediction:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}%")
    
