import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import torch

# Load model once
@st.cache_resource
def load_model():
    model = YOLO("best.pt")        # your trained classification model
    return model

model = load_model()

st.title("♻️ Waste Classification using YOLOv8")
st.write("Upload an image and the model will predict what type of waste it is.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    # Display image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Run prediction
    results = model.predict(img, verbose=False)
    r = results[0]

    # Extract predicted class and confidence
    probs = r.probs.data.cpu().numpy()
    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx] * 100)

    # YOLO stores classnames inside model.names
    class_name = model.names[pred_idx]

    st.subheader("Prediction Result")
    st.write(f"**Class:** {class_name}")
    st.write(f"**Confidence:** {confidence:.2f}%")
