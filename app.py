import os
import gdown
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("🐝 Bee vs. Fly Classifier 🪰")
st.write("Upload an image, and the AI model will predict whether it is a Bee or a Fly.")

@st.cache_resource
def load_model():
    file_id = '1iQSzUZMZ4jl_PA5_NMccEE7_2T3XdDL5'
    model_path = 'bee_vs_fly_model.keras'
    
        if not os.path.exists(model_path):
        url = f'https://drive.google.com/uc?id={file_id}&export=download'
        gdown.download(url, model_path, quiet=False, fuzzy=True)
            
        
    return tf.keras.models.load_model(model_path)

model = load_model()

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=300)
    
    if st.button("Predict"):
        img = image.resize((150, 150))
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(np.array(img, dtype=np.float32))
        img_array = np.expand_dims(img_array, axis=0)
        

        raw_pred = model.predict(img_array)[0][0]
        
        st.write(f"Raw Model Output: `{raw_pred}`")

        if raw_pred > 0.5:
            st.success(f"Result: **Fly 🪰** (Confidence: {raw_pred * 100:.1f}%)")
        else:
            st.success(f"Result: **Bee 🐝** (Confidence: {(1 - raw_pred) * 100:.1f}%)")
            
