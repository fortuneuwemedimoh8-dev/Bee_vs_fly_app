import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


# 1. Page Configuration & Header
st.set_page_config(
    page_title="Bee vs Fly Classifier - CE 11", page_icon="🐝", layout="centered"
)

st.title("🐝 Bee vs Fly Image Classifier")
st.subheader("CE 11 Mini Project")
st.write(
    "Upload an image of a bee or a fly to predict its class and view model confidence."
)

st.markdown("---")

# 2. Load the Saved Model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("bee_vs_fly_model(1).keras")
    return model
    


model = load_model()

# 3. Image Input Section
uploaded_file = st.file_uploader(
    "Choose an image (JPG, JPEG, PNG)...", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
  # Display uploaded image
  image = Image.open(uploaded_file).convert("RGB")
  st.image(image, caption="Uploaded Image", use_column_width=True)

  # 4. Prediction Button
  if st.button("🔍 Predict Image Class"):
    with st.spinner("Analyzing image..."):
      # Preprocess image to match training input (160×160)
      img_resized = image.resize((160, 160))
      img_array = np.array(img_resized)
      img_array = np.expand_dims(img_array, axis=0)  # Make batch shape (1, 160, 160, 3)

      # Make Prediction
      prediction = model.predict(img_array)[0][0]

      # Interpret Result (Sigmoid output: > 0.5 is Fly, <= 0.5 is Bee)
      if prediction > 0.5:
        label = "Fly 🪰"
        confidence = prediction * 100
      else:
        label = "Bee 🐝"
        confidence = (1 - prediction) * 100

      st.markdown("---")
      # 5. Display Prediction Results & Confidence Score
      st.success(f"**Prediction:** {label}")
      st.info(f"**Confidence Score:** {confidence:.2f}%")
        
