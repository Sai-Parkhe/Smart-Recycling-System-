import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Smart Waste AI", layout="wide")

# ---------------- CUSTOM UI ----------------
st.markdown("""
<style>
body {
    background-color: #f5f7fa;
}
.card {
    padding: 1.2rem;
    border-radius: 15px;
    background: white;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
    margin-bottom: 1rem;
}
.waste-badge {
    font-size: 22px;
    font-weight: 700;
    padding: 8px 18px;
    border-radius: 20px;
    display: inline-block;
    background: linear-gradient(135deg,#4CAF50,#2E8B57);
    color: white;
}
.center {
    text-align:center;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_trained_model():
    model = load_model("trash_classifier_mobilenetv2_part2.keras")
    labels = np.load("trash_classifier_part2_labels.npy", allow_pickle=True)
    return model, labels

model, class_labels = load_trained_model()

# ---------------- LOAD COMPANIES ----------------
@st.cache_data
def load_companies():
    df = pd.read_excel("companies.xlsx")   # keep file in same folder
    df["Waste_Needed"] = df["Waste_Needed"].str.lower().str.strip()
    return df

companies_df = load_companies()

# ---------------- TITLE ----------------
st.markdown("<h1 class='center'>♻ Smart Waste Classification & Recycling System</h1>", unsafe_allow_html=True)
st.markdown("<p class='center'>Upload or take a photo to classify waste and get recycling company recommendations.</p>", unsafe_allow_html=True)

# ---------------- IMAGE INPUT ----------------
col1, col2 = st.columns(2)
uploaded_image = None

with col1:
    upload = st.file_uploader("📁 Upload Image", type=["jpg", "jpeg", "png"])
    if upload:
        uploaded_image = Image.open(upload)

with col2:
    camera = st.camera_input("📷 Take Photo")
    if camera:
        uploaded_image = Image.open(camera)

# ---------------- PREDICTION ----------------
if uploaded_image:
    col_img, col_pred = st.columns([1,1])

    with col_img:
        st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

    img = uploaded_image.resize((224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    prediction = model.predict(img_array)
    pred_index = np.argmax(prediction)
    predicted_class = class_labels[pred_index]
    confidence = float(prediction[0][pred_index])

    with col_pred:
        st.subheader("🔍 AI Prediction")

        st.markdown(f"""
        <div class="card">
            <div class="waste-badge">🗂 {predicted_class.upper()}</div>
            <br><br>
            <b>Confidence:</b> {confidence*100:.2f}%
        </div>
        """, unsafe_allow_html=True)

        st.progress(confidence)

    # ---------------- FILTER COMPANIES ----------------
    matched = companies_df[
        companies_df["Waste_Needed"] == predicted_class.lower()
    ]

    st.subheader("🏭 Recommended Recycling Companies")

    if matched.empty:
        st.warning("No recycling companies found for this waste type.")
    else:
        for _, row in matched.iterrows():
            st.markdown(f"""
            <div class="card">
                <h4>🏢 {row['Company_Names']}</h4>
                📍 {row['Location']}<br>
                🌐 <a href="{row['Website']}" target="_blank">{row['Website']}</a>
            </div>
            """, unsafe_allow_html=True)
