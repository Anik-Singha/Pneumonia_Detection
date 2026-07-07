# ==========================================================
# Pneumonia Detection using ResNet50
# Streamlit Web Application
# ==========================================================

# ==========================================================
# Imports
# ==========================================================
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np


# ==========================================================
# Page Configuration
# ==========================================================
st.set_page_config(
    page_title="Pneumonia Detection",
    page_icon="🫁",
    layout="wide"
)
# ==========================================================
# Custom CSS
# ==========================================================
st.markdown("""
<style>
.block-container{
    padding-top:1.5rem;
    padding-bottom:1rem;
}
</style>
""", unsafe_allow_html=True)


# ==========================================================
# Title
# ==========================================================
st.markdown(
    "<h1 style='text-align:center;'>🫁 Chest X-ray Pneumonia Detection</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center;color:gray;'>Powered by ResNet50 Transfer Learning</p>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center;'>Upload a chest X-ray image and click <b>Predict</b> to classify it as NORMAL or PNEUMONIA.</p>",
    unsafe_allow_html=True
)

st.divider()


# ==========================================================
# Load Model
# ==========================================================
@st.cache_resource
def load_model():
    return keras.models.load_model("models/resnet50_pneumonia.keras")


model = load_model()


# ==========================================================
# Class Names
# ==========================================================
class_names = ["NORMAL", "PNEUMONIA"]


# ==========================================================
# Main Layout
# ==========================================================
info_col, image_col, result_col = st.columns([1, 2, 1.3])


# ==========================================================
# Left Column
# ==========================================================
with info_col:

    st.subheader("📋 Project Information")

    st.markdown("""
### 🧠 Model
ResNet50 (Transfer Learning)

---

### 📂 Dataset
Chest X-ray Images

---

### 🫁 Classes
- NORMAL
- PNEUMONIA

---

### 🖼 Image Size
224 × 224

---

### 🎯 Test Accuracy
95.38%

---

### ⚙ Framework
- TensorFlow
- Streamlit
- NumPy
- Pillow
""")


# ==========================================================
# Middle Column
# ==========================================================
with image_col:

    uploaded_file = st.file_uploader(
        "Upload Chest X-ray",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        st.image(
            image,
            caption="Uploaded Chest X-ray",
            use_container_width=True
        )


# ==========================================================
# Right Column
# ==========================================================
with result_col:

    st.subheader("🧠 Prediction")

    if uploaded_file is not None:

        if st.button("Predict", use_container_width=True):

            with st.spinner("Analyzing Chest X-ray..."):

                # ----------------------------------------
                # Image Preprocessing
                # ----------------------------------------
                image = image.convert("RGB")

                image = image.resize((224, 224))

                img = np.array(image)

                img = np.expand_dims(img, axis=0)

                img = tf.keras.applications.resnet.preprocess_input(img)

                # ----------------------------------------
                # Prediction
                # ----------------------------------------
                prediction = model.predict(img, verbose=0)

                predicted_class = np.argmax(prediction)

                confidence = float(np.max(prediction))

                # ----------------------------------------
                # Result
                # ----------------------------------------
                if predicted_class == 0:
                    st.success("✅ NORMAL")
                else:
                    st.error("🫁 PNEUMONIA")

                st.divider()

                st.subheader("Confidence")

                st.progress(confidence)

                st.write(f"### {confidence*100:.2f}%")

                st.divider()

                # ----------------------------------------
                # Probability of Each Class
                # ----------------------------------------
                st.subheader("Class Probabilities")

                normal_prob = float(prediction[0][0])
                pneumonia_prob = float(prediction[0][1])

                st.write("NORMAL")

                st.progress(normal_prob)

                st.write(f"{normal_prob*100:.2f}%")

                st.write("PNEUMONIA")

                st.progress(pneumonia_prob)

                st.write(f"{pneumonia_prob*100:.2f}%")


# ==========================================================
# Footer
# ==========================================================
st.divider()

st.caption(
    "Developed by Anik Singha | Deep Learning Project using ResNet50 and Streamlit"
)