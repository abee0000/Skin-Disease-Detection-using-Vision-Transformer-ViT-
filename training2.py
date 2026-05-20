import os
import tensorflow as tf
from transformers import ViTFeatureExtractor, TFViTForImageClassification, create_optimizer
from sklearn.metrics import classification_report
import numpy as np

# --- CONFIGURATION ---
dataset_dir = r"C:\Users\anoop\Desktop\SKIN DESEASE\SKIN_DESEASE\img1"  # Path to image dataset
img_size = (224, 224)  # Resize all images to this resolution
batch_size = 32
epochs = 20
split_ratio = 0.2  # 80% train, 20% validation

# --- LOAD DATASET ---
train_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    validation_split=split_ratio,
    subset="training",
    seed=42,
    image_size=img_size,
    batch_size=batch_size,
    label_mode="int"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    validation_split=split_ratio,
    subset="validation",
    seed=42,
    image_size=img_size,
    batch_size=batch_size,
    label_mode="int"
)

class_names = train_ds.class_names
num_classes = len(class_names)

# --- PREPROCESSING ---
AUTOTUNE = tf.data.AUTOTUNE

def preprocess(images, labels):
    images = tf.cast(images, tf.float32) / 255.0  # Normalize to [0, 1]
    images = tf.transpose(images, perm=[0, 3, 1, 2])  # ViT expects (B, C, H, W)
    return images, labels

train_ds = train_ds.map(preprocess).prefetch(AUTOTUNE)
val_ds = val_ds.map(preprocess).prefetch(AUTOTUNE)

# --- LOAD ViT MODEL ---
model = TFViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=num_classes,
    ignore_mismatched_sizes=True  # Adjust final classifier layer
)

# --- COMPILE MODEL ---
optimizer, schedule = create_optimizer(
    init_lr=2e-5,
    num_train_steps=len(train_ds) * epochs,
    num_warmup_steps=0
)

model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

# --- METRICS CALLBACK (F1, Precision, Recall) ---
class MetricsCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        y_true, y_pred = [], []
        for images, labels in val_ds:
            logits = self.model(images, training=False).logits
            preds = tf.argmax(logits, axis=-1)
            y_true.extend(labels.numpy())
            y_pred.extend(preds.numpy())
        report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
        print("\n" + report)

# --- EARLY STOPPING IF ACCURACY IN TARGET RANGE ---
class CustomEarlyStopping(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        acc = logs.get("accuracy")
        val_acc = logs.get("val_accuracy")
        if acc and val_acc:
            if 0.90 <= acc <= 0.99 and 0.90 <= val_acc <= 0.99:
                print(f"\n🛑 Early stopping: accuracy={acc:.4f}, val_accuracy={val_acc:.4f}")
                self.model.stop_training = True

# --- CHECKPOINT CALLBACK TO SAVE BEST MODEL ---
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    "best_model_002",
    monitor="val_accuracy",
    save_best_only=True,
    save_weights_only=False,
    mode="max",
    verbose=1
)

# --- PRINT MODEL SUMMARY ---
model.summary()

# --- TRAIN MODEL ---
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=[MetricsCallback(), checkpoint_cb, CustomEarlyStopping()]
)

# --- SAVE FINAL MODEL ---
model.save_pretrained("final_model_002")
print("\n✅ Final model saved in 'final_model_002'")
