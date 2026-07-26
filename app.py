import numpy as np
import streamlit as st
import tensorflow as tf
from huggingface_hub import hf_hub_download
from PIL import Image

st.title("Bee vs Fly Classifier 🐝🪰")
st.write("Upload an image to test whether it's a Bee or a Fly!")


# Download and cache model directly from Hugging Face
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

    # Preprocess
    img_resized = image.resize((150, 150))
    img_array = np.array(img_resized)
    img_batch = np.expand_dims(img_array, axis=0)

    # Predict logic
    predictions = model.predict(img_batch)[0]

    if len(predictions) == 1:
        score = float(predictions[0])
        bee_prob = 1.0 - score
        fly_prob = score
    else:
        bee_prob = float(predictions[0])
        fly_prob = float(predictions[1])

    # Display Results
    st.write("### Predictions:")
    st.progress(bee_prob, text=f"Bee: {bee_prob * 100:.2f}%")
    st.progress(fly_prob, text=f"Fly: {fly_prob * 100:.2f}%")
    
