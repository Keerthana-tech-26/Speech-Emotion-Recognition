import os
import numpy as np
from utils import extract_mfcc
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Reshape, LSTM, Dense
from keras.optimizers import Adam

# Emotion labels
emotions = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}
emotion_labels = sorted(list(set(emotions.values())))

# Dataset path
DATA_PATH = os.path.join("dataset", "RAVDESS")

X, y = [], []

# Limit files for faster testing
file_count = 0
max_files = 200  # Adjust this as needed

for root, dirs, files in os.walk(DATA_PATH):
    for file in files:
        if file.endswith('.wav'):
            if file_count >= max_files:
                break
            emotion_code = file.split('-')[2]
            label = emotions.get(emotion_code)
            if label:
                file_path = os.path.join(root, file)
                mfcc = extract_mfcc(file_path)
                if mfcc is not None:
                    X.append(mfcc)
                    y.append(emotion_labels.index(label))
                    file_count += 1

print(f"✅ Total files processed: {file_count}")

X = np.array(X)
X = X.reshape(X.shape[0], 40, 174, 1)
y = to_categorical(np.array(y))

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# CNN + LSTM Model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(40, 174, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))
model.add(Reshape((19, 2752)))
model.add(LSTM(64))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(len(emotion_labels), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])
model.summary()

print("🚀 Starting training...")
model.fit(X_train, y_train, epochs=5, batch_size=16, validation_data=(X_test, y_test))

# Save model
os.makedirs("model", exist_ok=True)
model.save("model/cnn_lstm_model.h5")
print("✅ Model saved to model/cnn_lstm_model.h5")
