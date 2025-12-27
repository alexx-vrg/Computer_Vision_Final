import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
    top_k_accuracy_score,
)

# =================CONFIGURATION=================
# Define paths to your data set folders (flat structure with CSV labels)
TRAIN_PATH = './train'
TEST_PATH = './test'
LABELS_CSV = './train_labels.csv'

# Define image parameters used for training
BATCH_SIZE = 32
IMG_HEIGHT = 128  # Resize all images to this standard size
IMG_WIDTH = 128
EPOCHS = 15       # How many times the model sees the whole dataset

print(f"TensorFlow Version: {tf.__version__}")
print(f"Checking for train data at: {TRAIN_PATH}")

# ================= 1. DATA LOADING =================

# Load labels from CSV
labels_df = pd.read_csv(LABELS_CSV)
print(f"Loaded {len(labels_df)} labels")

# Get class names
class_names = sorted(labels_df['Label'].unique().tolist())
num_classes = len(class_names)
print(f"\nDetected classes: {class_names}")

# Create label to index mapping
label_to_index = {label: idx for idx, label in enumerate(class_names)}

# Create a dictionary mapping image ID to label index
id_to_label = dict(zip(labels_df['Id'].astype(str).str.zfill(5), labels_df['Label']))

# Get all training image paths
train_image_paths = sorted([str(p) for p in pathlib.Path(TRAIN_PATH).glob('*.png')])
print(f"Found {len(train_image_paths)} training images")

# Create labels array
train_labels = []
valid_paths = []
for path in train_image_paths:
    img_id = os.path.splitext(os.path.basename(path))[0]
    if img_id in id_to_label:
        train_labels.append(label_to_index[id_to_label[img_id]])
        valid_paths.append(path)

train_labels = np.array(train_labels)
train_image_paths = valid_paths
print(f"Matched {len(train_labels)} images with labels")

# Function to load and preprocess image
def load_and_preprocess_image(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    img = img / 255.0  # Normalize to [0,1]
    return img, label

# Create TensorFlow dataset
train_ds_full = tf.data.Dataset.from_tensor_slices((train_image_paths, train_labels))

# Split into train and validation (80/20)
total_size = len(train_image_paths)
val_size = int(0.2 * total_size)
train_size = total_size - val_size

train_ds_full = train_ds_full.shuffle(buffer_size=total_size, seed=123)
train_ds = train_ds_full.skip(val_size)
val_ds = train_ds_full.take(val_size)

# Apply preprocessing
train_ds = train_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

# Batch and prefetch for performance
train_ds = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

print(f"Training batches: {len(list(train_ds))}")
print(f"Validation batches: {len(list(val_ds))}")

# Reload datasets (iteration consumed them)
train_ds = tf.data.Dataset.from_tensor_slices((train_image_paths, train_labels))
train_ds = train_ds.shuffle(buffer_size=total_size, seed=123)
train_ds_split = train_ds.skip(val_size)
val_ds_split = train_ds.take(val_size)
train_ds = train_ds_split.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds_split.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# ================= 2. BUILD THE CNN MODEL =================

# This is a basic Convolutional Neural Network architecture.
model = Sequential([
    # Input layer
    layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    
    # Convolutional layers find features (edges, shapes, colors)
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    
    # Flatten turns 2D feature maps into a 1D list
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    
    # The final layer decides which class it belongs to
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()


# ================= 3. TRAIN THE MODEL =================
print("\nStarting Training...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)
print("Training finished.")

# ================= VALIDATION METRICS (PER CLASS) =================
print("\nValidation metrics per class:")
y_true = []
y_pred = []
all_probs = []
for images, labels in val_ds:
    preds = model.predict(images, verbose=0)
    y_true.append(labels.numpy())
    y_pred.append(np.argmax(preds, axis=1))
    all_probs.append(preds)
y_true = np.concatenate(y_true)
y_pred = np.concatenate(y_pred)
all_probs = np.concatenate(all_probs)
report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
print(report)

# Additional summary stats
overall_acc = accuracy_score(y_true, y_pred)
macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
    y_true, y_pred, average="macro", zero_division=0
)
weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
    y_true, y_pred, average="weighted", zero_division=0
)
top3_acc = top_k_accuracy_score(y_true, all_probs, k=3, labels=range(num_classes))

print("Overall metrics:")
print(f"  Accuracy (top-1): {overall_acc:.4f}")
print(f"  Accuracy (top-3): {top3_acc:.4f}")
print(f"  Macro  P/R/F1   : {macro_p:.4f} / {macro_r:.4f} / {macro_f1:.4f}")
print(f"  Weighted P/R/F1 : {weighted_p:.4f} / {weighted_r:.4f} / {weighted_f1:.4f}")

# Confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
print("\nConfusion matrix (rows=true, cols=pred):")
print(cm_df)


# ================= 4. PREDICTION ON TEST DATA =================
print("\nStarting predictions on test data...")

test_data_dir = pathlib.Path(TEST_PATH)
test_image_paths = sorted([str(p) for p in test_data_dir.glob('*.png')])

if not test_image_paths:
    print("Error: No PNG files found in the test directory!")
    exit()

print(f"Found {len(test_image_paths)} images to predict.")

# Extract IDs from filenames
submission_ids = [os.path.splitext(os.path.basename(p))[0] for p in test_image_paths]

# Function to load test image (no label needed)
def load_test_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    img = img / 255.0
    return img

# Create batched test dataset for fast prediction
test_ds = tf.data.Dataset.from_tensor_slices(test_image_paths)
test_ds = test_ds.map(load_test_image, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Predict all at once
print("Running batched predictions...")
predictions_array = model.predict(test_ds, verbose=1)

# Convert predictions to labels
predicted_indices = np.argmax(predictions_array, axis=1)
predictions = [class_names[idx] for idx in predicted_indices]
print(f"Completed {len(predictions)} predictions.")

# ================= 5. CREATE SUBMISSION.CSV =================

submission_df = pd.DataFrame({
    'Id': submission_ids,
    'Label': predictions
})

output_filename = "submission_tensorflow.csv"
submission_df.to_csv(output_filename, index=False)

print(f"\nSuccessfully created {output_filename}!")
print(submission_df.head(10))
