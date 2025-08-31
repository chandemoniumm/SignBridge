import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Parameters
SEQ_LENGTH = 30  # Number of frames per sequence
TOP_N_CLASSES = 10  # Number of most frequent classes to keep

# Load dataset
data = pd.read_csv('/home/chandemonium/isl_project/code/isl_pose_hand_landmarks.csv')

# Extract feature columns (excluding 'label', 'video', 'frame')
feature_cols = [col for col in data.columns if col not in ['label', 'video', 'frame']]

# Group frames by 'label' and 'video', build sequences
sequences = []
labels = []

for (label, video), group in data.groupby(['label', 'video']):
    group_sorted = group.sort_values('frame')
    if len(group_sorted) >= SEQ_LENGTH:
        frames = group_sorted.iloc[:SEQ_LENGTH][feature_cols].values
        sequences.append(frames)
        labels.append(label)

X = np.array(sequences)
y = np.array(labels)

# Filter dataset to keep only top N most frequent classes
class_counts = Counter(labels)
top_classes = [label for label, _ in class_counts.most_common(TOP_N_CLASSES)]

filtered_indices = [i for i, lbl in enumerate(y) if lbl in top_classes]
X = X[filtered_indices]
y = y[filtered_indices]

print(f"Filtered dataset to top {TOP_N_CLASSES} classes. Remaining samples: {len(y)}")

# Encode string labels to integers
label_encoder = LabelEncoder()
y_int = label_encoder.fit_transform(y)

num_classes = len(np.unique(y_int))
print(f"Number of classes after filtering: {num_classes}")

# One-hot encode labels
y_cat = to_categorical(y_int, num_classes=num_classes)

# Train-test split without stratify to avoid errors
X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, random_state=42)

features_per_frame = X.shape[2]

# Build model with stacked LSTM and dropout
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(SEQ_LENGTH, features_per_frame)))
model.add(Dropout(0.3))
model.add(LSTM(64))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# Callbacks for early stopping and saving best model
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint('/home/chandemonium/isl_project/code/best_isl_lstm_model.h5',
                             monitor='val_accuracy', save_best_only=True, verbose=1)

# Train model
model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop, checkpoint]
)

# Evaluate best model on test set
y_pred_prob = model.predict(X_test)
y_pred_int = np.argmax(y_pred_prob, axis=1)
y_true_int = np.argmax(y_test, axis=1)

y_pred_labels = label_encoder.inverse_transform(y_pred_int)
y_true_labels = label_encoder.inverse_transform(y_true_int)

print(f"Test Accuracy: {accuracy_score(y_true_labels, y_pred_labels):.4f}")
print("Classification Report:")
print(classification_report(y_true_labels, y_pred_labels))
