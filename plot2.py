import os
import tensorflow as tf
from transformers import TFViTForImageClassification
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIG ---
dataset_dir = r"D:\Verdant\SKIN_DESEASES\img1"
img_size = (224, 224)
batch_size = 32
split_ratio = 0.2
verbose = True  # Set to True to print predictions batch-wise

# --- LOAD VALIDATION SET ---
val_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    validation_split=split_ratio,
    subset="validation",
    seed=42,
    image_size=img_size,
    batch_size=batch_size,
    label_mode="int"
)

class_names = val_ds.class_names
num_classes = len(class_names)

# --- PREPROCESS ---
def preprocess(images, labels):
    images = tf.cast(images, tf.float32) / 255.0
    images = tf.transpose(images, perm=[0, 3, 1, 2])  # (B, H, W, C) → (B, C, H, W)
    return images, labels

AUTOTUNE = tf.data.AUTOTUNE
val_ds = val_ds.map(preprocess).prefetch(AUTOTUNE)

# --- LOAD SAVED MODEL ---
print("🔄 Loading model...")
model = TFViTForImageClassification.from_pretrained("final_model_001")
print("✅ Model loaded.\n")

# --- PRINT MODEL SUMMARY ---
print("🧠 Model Summary:\n")
dummy_input = tf.random.normal((1, 3, 224, 224))
model(dummy_input)  # Build the model
model.summary()

# --- PREDICT & EVALUATE ---
y_true, y_pred = [], []
print("\n🔍 Running predictions...\n")
for batch_index, (images, labels) in enumerate(val_ds):
    logits = model(images, training=False).logits
    preds = tf.argmax(logits, axis=-1)
    y_true.extend(labels.numpy())
    y_pred.extend(preds.numpy())

    if verbose:
        print(f"📦 Batch {batch_index+1}:")
        for i in range(len(labels)):
            true_label = class_names[labels[i].numpy()]
            pred_label = class_names[preds[i].numpy()]
            print(f"    - True: {true_label:25} | Predicted: {pred_label}")
        print()

# --- CLASSIFICATION REPORT ---
print("\n📋 Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

# --- CONFUSION MATRIX ---
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("🧾 Confusion Matrix")
plt.tight_layout()
plt.savefig("final_confusion_matrix.png")
plt.show()
