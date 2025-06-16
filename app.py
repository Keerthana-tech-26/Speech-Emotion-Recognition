import streamlit as st
import joblib
import numpy as np
from utils import extract_features

st.title("🎤 Speech Emotion Recognition")

uploaded_file = st.file_uploader("Upload an audio file (.wav)", type=["wav"])

if uploaded_file:
    st.audio(uploaded_file, format="audio/wav")
    
    # Save file temporarily
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())

    features = extract_features("temp.wav").reshape(1, -1)
    model = joblib.load("model.pkl")
    prediction = model.predict(features)
    proba = model.predict_proba(features)

    st.write(f"**Predicted Emotion:** `{prediction[0].upper()}`")
    st.bar_chart(proba[0])
