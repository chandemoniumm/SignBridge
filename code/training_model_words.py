
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# Parameters
SEQ_LENGTH = 30  # Number of frames per sequence; adjust per your data
TOP_N_CLASSES = 10  # Optional: focus on most common classes to improve learning

# Load your word-level CSV data
data_path = '/home/chandemonium/isl_project/code/isl_pose_hand_landmarks_word_level.csv'
df = pd.read_csv(data_path)

# Extract feature columns (all pose_ and hand_ columns)
feature_cols = [col for col in df.columns if col.startswith('pose_') or col.startswith('hand_')]

# Group by 'label' and 'video' to create sequences of frames
sequences = []
labels = []

for (label, video), group in df.groupby(['label', 'video']):
    group_sorted = group.sort_values('frame')
    frames = group_sorted[feature_cols].values
    # Slide over frames to form sequences of length SEQ_LENGTH
    for start_idx in range(0, len(frames) - SEQ_LENGTH + 1, SEQ_LENGTH):
        seq = frames[start_idx:start_idx + SEQ_LENGTH]
        sequences.append(seq)
        labels.append(label)

X = np.array(sequences)
y = np.array(labels)

# Keep only top N classes for more balanced training (optional)
if TOP_N_CLASSES:
    class_counts = Counter(labels)
    top_classes = [c for c, _ in class_counts.most_common(TOP_N_CLASSES)]
    filtered_indices = [i for i, lbl in enumerate(y) if lbl in top_classes]
    X = X[filtered_indices]
    y = y[filtered_indices]

# Normalize features (fit on entire dataset)
num_samples, seq_len, feat_dim = X.shape
X_reshaped = X.reshape(-1, feat_dim)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_reshaped)
X = X_scaled.reshape(num_samples, seq_len, feat_dim)

# Encode labels
label_encoder = LabelEncoder()
y_int = label_encoder.fit_transform(y)
num_classes = len(np.unique(y_int))
y_cat = to_categorical(y_int, num_classes=num_classes)

print(f"Number of classes: {num_classes}")
print(f"Dataset shape: {X.shape}")

# Split train-test without stratify (avoid errors if rare classes)
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

# Build LSTM model
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(SEQ_LENGTH, feat_dim)))
model.add(Dropout(0.3))
model.add(LSTM(64))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train model
model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop]
)

# Evaluate on test set
y_pred_prob = model.predict(X_test)
y_pred_int = np.argmax(y_pred_prob, axis=1)
y_true_int = np.argmax(y_test, axis=1)

y_pred_labels = label_encoder.inverse_transform(y_pred_int)
y_true_labels = label_encoder.inverse_transform(y_true_int)

print(f"Test Accuracy: {accuracy_score(y_true_labels, y_pred_labels):.4f}")
print("Classification Report:")
print(classification_report(y_true_labels, y_pred_labels))
