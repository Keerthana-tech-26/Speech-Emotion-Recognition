import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import pandas as pd
from utils import extract_mfcc
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

# Load trained model
model = tf.keras.models.load_model("model/cnn_lstm_model.h5")

# Emotion labels
emotions = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}
emotion_labels = sorted(list(set(emotions.values())))

# Streamlit UI
st.set_page_config(page_title="Speech Emotion Recognition", page_icon="🎙️")
st.title("🎙️ Speech Emotion Recognition App")
st.markdown("Upload a `.wav` file to detect the speaker's emotion using a deep learning model (CNN + LSTM).")

# File uploader
uploaded_file = st.file_uploader("Upload Audio File (.wav only)", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    
    # Save the uploaded file temporarily
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())
    
    # Extract features and predict
    mfcc = extract_mfcc("temp.wav")
    if mfcc is not None:
        mfcc = mfcc.reshape(1, 40, 174, 1)
        prediction = model.predict(mfcc)
        predicted_label = emotion_labels[np.argmax(prediction)]
        
        st.success(f"🎯 Predicted Emotion: **{predicted_label.upper()}**")

        # Display prediction confidence as a bar chart
        st.subheader("📊 Prediction Confidence:")
        probs = prediction[0]
        df_probs = pd.DataFrame({
            "Emotion": emotion_labels,
            "Probability": probs
        })

        st.bar_chart(df_probs.set_index("Emotion"))
st.markdown("---")
st.subheader("📈 Model Evaluation (on test data sample)")

DATA_PATH = os.path.join("dataset", "RAVDESS")
X_eval, y_eval = [], []
file_count = 0
max_files = 200  # same as used during training

for root, dirs, files in os.walk(DATA_PATH):
    for file in files:
        if file.endswith('.wav') and file_count < max_files:
            emotion_code = file.split('-')[2]
            label = emotions.get(emotion_code)
            if label:
                file_path = os.path.join(root, file)
                mfcc = extract_mfcc(file_path)
                if mfcc is not None:
                    X_eval.append(mfcc)
                    y_eval.append(emotion_labels.index(label))
                    file_count += 1

# Process evaluation
if X_eval and y_eval:
    X_eval = np.array(X_eval).reshape(-1, 40, 174, 1)
    y_eval = np.array(y_eval)

    _, X_test, _, y_test = train_test_split(X_eval, y_eval, test_size=0.2, random_state=42)

    y_pred = model.predict(X_test)
    y_pred_labels = np.argmax(y_pred, axis=1)

    accuracy = accuracy_score(y_test, y_pred_labels)
    report_text = classification_report(y_test, y_pred_labels, target_names=emotion_labels)

    st.write(f"✅ **Accuracy on test data**: `{accuracy:.2f}`")
    st.text("📊 Classification Report:")
    st.text(report_text)
else:
    st.warning("⚠️ Not enough data loaded to evaluate model.")
