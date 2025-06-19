import streamlit as st
import numpy as np
import tensorflow as tf
import os
import pandas as pd
from utils import extract_mfcc
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

model = tf.keras.models.load_model("model/cnn_lstm_model.h5")
emotion_labels = sorted(['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised'])

st.set_page_config(page_title="🎙️ Speech Emotion Recognition", page_icon="🎧", layout="centered")
st.markdown("<h1 style='text-align: center; color: #ff4b4b;'>🎙️ TESS Speech Emotion Recognition</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Upload a .wav file and detect emotion using CNN + LSTM</h4>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📤 Upload Audio File (.wav)", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())
    mfcc = extract_mfcc("temp.wav")
    if mfcc is not None:
        mfcc_input = mfcc.reshape(1, 40, 174, 1)
        prediction = model.predict(mfcc_input)
        predicted_index = np.argmax(prediction)
        predicted_emotion = emotion_labels[predicted_index]

        st.markdown(f"<h2 style='color:#00bfff;'>🎯 Predicted Emotion: <span style='color:#ff4b4b'>{predicted_emotion.upper()}</span></h2>", unsafe_allow_html=True)

        df_probs = pd.DataFrame({
            'Emotion': emotion_labels,
            'Probability': prediction[0]
        })
        st.subheader("📊 Prediction Confidence")
        st.bar_chart(df_probs.set_index("Emotion"))

st.markdown("---")
st.subheader("📈 Model Evaluation (on sample TESS data)")

DATA_PATH = os.path.join("dataset", "TESS")
X_eval, y_eval = [], []
file_count = 0
max_files = 200

for folder in os.listdir(DATA_PATH):
    folder_path = os.path.join(DATA_PATH, folder)
    if not os.path.isdir(folder_path) or not folder.startswith(('OAF', 'YAF')):
        continue
    label = folder.split('_')[-1].lower()
    for file in os.listdir(folder_path):
        if file.endswith(".wav") and file_count < max_files:
            file_path = os.path.join(folder_path, file)
            mfcc = extract_mfcc(file_path)
            if mfcc is not None:
                X_eval.append(mfcc)
                y_eval.append(emotion_labels.index(label))
                file_count += 1

if X_eval and y_eval:
    X_eval = np.array(X_eval).reshape(-1, 40, 174, 1)
    y_eval = np.array(y_eval)
    _, X_test, _, y_test = train_test_split(X_eval, y_eval, test_size=0.2, random_state=42)
    y_pred = model.predict(X_test)
    y_pred_labels = np.argmax(y_pred, axis=1)
    accuracy = accuracy_score(y_test, y_pred_labels)
    report_text = classification_report(
        y_test,
        y_pred_labels,
        labels=range(len(emotion_labels)),
        target_names=emotion_labels,
        zero_division=0
    )
    st.markdown(f"<h4 style='color:#28a745;'>✅ Accuracy on test data: {accuracy:.2f}</h4>", unsafe_allow_html=True)
    st.text("📄 Classification Report:")
    st.text(report_text)
else:
    st.warning("⚠️ Not enough data available to display evaluation metrics.")
