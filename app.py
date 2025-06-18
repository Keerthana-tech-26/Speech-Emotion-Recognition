import streamlit as st
import numpy as np
import tensorflow as tf
from utils.extract_features import extract_mfcc

st.title("🎙️ Speech Emotion Recognition")

uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

emotion_labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

model = tf.keras.models.load_model("model/cnn_lstm_model.h5")

if uploaded_file:
    mfcc = extract_mfcc(uploaded_file)
    if mfcc is not None:
        input_data = mfcc.reshape(1, 40, 174, 1)
        prediction = model.predict(input_data)
        emotion = emotion_labels[np.argmax(prediction)]
        st.subheader(f"🧠 Predicted Emotion: **{emotion}**")
