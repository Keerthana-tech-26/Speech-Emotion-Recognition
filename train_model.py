import os
import numpy as np
import librosa
import soundfile
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# Emotions based on RAVDESS naming
emotions = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}

def extract_features(file):
    audio, sr = librosa.load(file, res_type='kaiser_fast')
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)

DATA_PATH = r'C:\Users\keert\OneDrive\Desktop\speech_emotion\dataset\RAVDESS'

X, y = [], []

for root, dirs, files in os.walk(DATA_PATH):
    for file in files:
        if file.endswith('.wav'):
            print(f"Processing file: {file}")  # ✅ Debug line
            emotion_code = file.split('-')[2]
            label = emotions.get(emotion_code)
            if label:
                file_path = os.path.join(root, file)
                try:
                    features = extract_features(file_path)
                    X.append(features)
                    y.append(label)
                except Exception as e:
                    print(f"Failed to process {file_path}: {e}")


print(f"✅ Total samples collected: {len(X)}")

X, y = np.array(X), np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)

joblib.dump(model, "model.pkl")
print("✅ Model trained and saved as model.pkl")
print("🎯 Accuracy:", accuracy_score(y_test, model.predict(X_test)))
