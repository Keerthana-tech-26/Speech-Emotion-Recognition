import os
import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LSTM, Reshape, TimeDistributed
from tensorflow.keras.callbacks import ModelCheckpoint
import warnings

warnings.filterwarnings("ignore")

DATA_PATH = "dataset/RAVDESS/audio_speech_actors_01-24"
SAMPLE_RATE = 22050
MAX_LEN = 174
EMOTIONS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        if mfcc.shape[1] < MAX_LEN:
            pad_width = MAX_LEN - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :MAX_LEN]
        return mfcc
    except Exception as e:
        print(f"Skipped {os.path.basename(file_path)} (MFCC extraction failed)")
        return None

def load_data(data_path):
    X, y = [], []
    for root, _, files in os.walk(data_path):
        for file in files:
            if file.endswith(".wav"):
                emotion_label = int(file.split("-")[2]) - 1  # 1-indexed in filename
                file_path = os.path.join(root, file)
                mfcc = extract_features(file_path)
                if mfcc is not None:
                    X.append(mfcc)
                    y.append(emotion_label)
    print(f"\n✅ Total usable samples: {len(X)}")
    if len(X) == 0:
        raise ValueError("\u274c No data loaded. Check your dataset path and audio files.")
    return np.array(X), to_categorical(y, num_classes=len(EMOTIONS))

print(f"Loading data from: {DATA_PATH}")
X, y = load_data(DATA_PATH)
X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(40, MAX_LEN, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.3))
model.add(TimeDistributed(Flatten()))
model.add(Reshape((model.output_shape[1], model.output_shape[2])))
model.add(LSTM(128, return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(len(EMOTIONS), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

os.makedirs("model", exist_ok=True)
checkpoint = ModelCheckpoint("model/cnn_lstm_model.h5", monitor='val_accuracy', save_best_only=True, mode='max')

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[checkpoint]
)

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
