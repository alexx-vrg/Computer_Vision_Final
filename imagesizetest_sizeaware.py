import numpy as np
import os
import pandas as pd
import pathlib
import random
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
    top_k_accuracy_score,
)

# ================= REPRODUCIBILITY =================
# Set seeds for Python, NumPy, and TensorFlow to ensure run-to-run reproducibility.
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Enforce deterministic ops where possible (may reduce performance slightly).
tf.config.experimental.enable_op_determinism()

# ================= CONFIGURATION =================
TRAIN_PATH = "./train"
TEST_PATH = "./test"
LABELS_CSV = "./train_labels.csv"

IMG_HEIGHT = 128  # canvas size (images are resized for CNN branch)
IMG_WIDTH = 128
BATCH_SIZE = 32
EPOCHS = 15

print(f"TensorFlow Version: {tf.__version__}")
print(f"Checking for train data at: {TRAIN_PATH}")


def ensure_paths():
    for p in [TRAIN_PATH, TEST_PATH, LABELS_CSV]:
        if not pathlib.Path(p).exists():
            raise FileNotFoundError(f"Required path not found: {p}")

# ================= 1. DATA LOADING =================
ensure_paths()
labels_df = pd.read_csv(LABELS_CSV)
print(f"Loaded {len(labels_df)} labels")

class_names = sorted(labels_df["Label"].unique().tolist())
num_classes = len(class_names)
print(f"\nDetected classes: {class_names}")

label_to_index = {label: idx for idx, label in enumerate(class_names)}
id_to_label = dict(zip(labels_df["Id"].astype(str).str.zfill(5), labels_df["Label"]))

train_image_paths = sorted([str(p) for p in pathlib.Path(TRAIN_PATH).glob("*.png")])
print(f"Found {len(train_image_paths)} training images")

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


def load_and_preprocess_with_size(path, label):
    """Load image, keep original size as feature, resize for CNN."""
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_png(img_bytes, channels=3)

    orig_shape = tf.cast(tf.shape(img)[:2], tf.float32)  # (h, w)
    size_feat = orig_shape / 256.0  # simple normalization

    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    img = img / 255.0
    return (img, size_feat), label


# Build tf.data
ds_full = tf.data.Dataset.from_tensor_slices((train_image_paths, train_labels))

total_size = len(train_image_paths)
val_size = int(0.2 * total_size)

ds_full = ds_full.shuffle(buffer_size=total_size, seed=123)
train_ds = ds_full.skip(val_size)
val_ds = ds_full.take(val_size)

train_ds = (
    train_ds.map(load_and_preprocess_with_size, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)
val_ds = (
    val_ds.map(load_and_preprocess_with_size, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

# ================= 2. BUILD THE CNN + SIZE BRANCH MODEL =================
img_input = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3), name="img")
size_input = layers.Input(shape=(2,), name="size")  # (h, w) normalized

x = layers.Conv2D(16, 3, padding="same", activation="relu")(img_input)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)
x = layers.Flatten()(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.3)(x)

# Concatenate size features
combined = layers.Concatenate()([x, size_input])
combined = layers.Dense(64, activation="relu")(combined)
combined = layers.Dropout(0.2)(combined)
output = layers.Dense(num_classes, activation="softmax")(combined)

model = Model(inputs=[img_input, size_input], outputs=output)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# ================= 3. TRAIN THE MODEL =================
print("\nStarting Training (size-aware)...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
)
print("Training finished.")

# ================= VALIDATION METRICS (PER CLASS) =================
print("\nValidation metrics per class:")
y_true = []
y_pred = []
all_probs = []
for (images, sizes), labels in val_ds:
    preds = model.predict([images, sizes], verbose=0)
    y_true.append(labels.numpy())
    y_pred.append(np.argmax(preds, axis=1))
    all_probs.append(preds)
y_true = np.concatenate(y_true)
y_pred = np.concatenate(y_pred)
all_probs = np.concatenate(all_probs)
report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
print(report)

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

cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
print("\nConfusion matrix (rows=true, cols=pred):")
print(cm_df)

# ================= 4. PREDICTION ON TEST DATA =================
print("\nStarting predictions on test data...")

test_image_paths = sorted([str(p) for p in pathlib.Path(TEST_PATH).glob("*.png")])
if not test_image_paths:
    print("Error: No PNG files found in the test directory!")
    exit()

print(f"Found {len(test_image_paths)} images to predict.")
submission_ids = [os.path.splitext(os.path.basename(p))[0] for p in test_image_paths]


def load_test_image_with_size(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_png(img_bytes, channels=3)
    orig_shape = tf.cast(tf.shape(img)[:2], tf.float32)
    size_feat = orig_shape / 256.0
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    img = img / 255.0
    return img, size_feat


test_ds = tf.data.Dataset.from_tensor_slices(test_image_paths)
test_ds = test_ds.map(load_test_image_with_size, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

print("Running batched predictions...")
imgs_batch = []
sizes_batch = []
for imgs, sizes in test_ds:
    imgs_batch.append(imgs)
    sizes_batch.append(sizes)
imgs_batch = tf.concat(imgs_batch, axis=0)
sizes_batch = tf.concat(sizes_batch, axis=0)

predictions_array = model.predict([imgs_batch, sizes_batch], verbose=1)
predicted_indices = np.argmax(predictions_array, axis=1)
predictions = [class_names[idx] for idx in predicted_indices]
print(f"Completed {len(predictions)} predictions.")

submission_df = pd.DataFrame(
    {
        "Id": submission_ids,
        "Label": predictions,
    }
)
output_filename = "submission_tensorflow_sizeaware.csv"
submission_df.to_csv(output_filename, index=False)

print(f"\nSuccessfully created {output_filename}!")
print(submission_df.head(10))

