
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="AI WasteSort",
    page_icon="♻️",
    layout="centered"
)

CLASSES = ["glass", "metal", "organic", "paper", "plastic"]

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "AI_WasteSort_v1.keras",
        compile=False
    )

st.title("♻️ AI WasteSort")
st.subheader("Smart Waste Classification")
st.write(
    "Upload a waste image or take a photo "
    "to predict its category using AI."
)

st.info(
    "Educational demonstration. Predictions may be incorrect; "
    "check items manually before sorting."
)

try:
    model = load_model()
    st.success("AI model loaded successfully!")
except Exception as e:
    st.error("The AI model could not be loaded.")
    st.caption(str(e))
    st.stop()

image_file = st.camera_input("Take a photo") 

if image_file is None:
    image_file = st.file_uploader(
        "Or upload a waste image",
        type=["jpg", "jpeg", "png"]
    )

if image_file is not None:
    image = Image.open(image_file).convert("RGB")
    st.image(image, caption="Selected waste image",
             use_container_width=True)

    if st.button("🔍 Classify Waste", type="primary"):
        with st.spinner("AI is analysing the image..."):
            image = image.resize((224, 224))
            array = np.array(image, dtype=np.float32)
            array = np.expand_dims(array, axis=0)

            predictions = model.predict(array, verbose=0)[0]
            index = int(np.argmax(predictions))

        st.subheader("Prediction")
        st.success(f"♻️ {CLASSES[index].title()}")

        st.write("Model scores")
        for name, score in zip(CLASSES, predictions):
            st.write(f"**{name.title()}:** {score * 100:.2f}%")
            st.progress(float(np.clip(score, 0, 1)))

        st.caption(
            "Scores are model outputs, not guaranteed probabilities "
            "of correctness."
          )
      
