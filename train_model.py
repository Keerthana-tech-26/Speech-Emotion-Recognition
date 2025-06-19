import os
import numpy as np
from utils import extract_mfcc
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Reshape, LSTM, Dense
from keras.optimizers import Adam

DATA_PATH = os.path.join("dataset", "TESS")
X, y = [], []

emotion_labels = sorted(list(set(
    folder.split('_')[-1].lower()
    for folder in os.listdir(DATA_PATH)
    if os.path.isdir(os.path.join(DATA_PATH, folder)) and folder.startswith(('OAF', 'YAF'))
)))
print("✅ Detected emotion labels:", emotion_labels)
print("🧠 Number of emotion classes:", len(emotion_labels))
for folder in os.listdir(DATA_PATH):
    folder_path = os.path.join(DATA_PATH, folder)
    if not os.path.isdir(folder_path) or not folder.startswith(('OAF', 'YAF')):
        continue
    label = folder.split('_')[-1].lower()
    for file in os.listdir(folder_path):
        if file.endswith('.wav'):
            file_path = os.path.join(folder_path, file)
            mfcc = extract_mfcc(file_path)
            if mfcc is not None:
                X.append(mfcc)
                y.append(emotion_labels.index(label))

print(f"\n🎧 Total audio samples collected: {len(X)}")

if len(X) == 0:
    print("❌ No valid audio data found. Check dataset path and folder names.")
    exit()

X = np.array(X)
X = X.reshape(X.shape[0], 40, 174, 1)
y = to_categorical(np.array(y), num_classes=len(emotion_labels))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

print("\n🚀 Starting model training...")
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

os.makedirs("model", exist_ok=True)
model.save("model/cnn_lstm_model.h5")
print("\n✅ Model saved to model/cnn_lstm_model.h5")
