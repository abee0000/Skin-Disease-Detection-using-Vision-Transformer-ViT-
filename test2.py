import tensorflow as tf
import numpy as np
from vit_keras import vit
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.losses import CategoricalCrossentropy

# ✅ Step 1: Define Model Path
MODEL_PATH = "vit_best_model_001.h5"

# ✅ Step 2: Register Custom Objects for ViT
custom_objects = get_custom_objects()
custom_objects.update({
    "ClassToken": vit.__dict__.get("ClassToken"),
    "TransformerBlock": vit.__dict__.get("TransformerBlock"),
    "TransformerEncoder": vit.__dict__.get("TransformerEncoder"),
    "ExtractToken": vit.__dict__.get("ExtractToken"),
    "loss": CategoricalCrossentropy()  # ✅ Fix: Explicitly add loss function
})

# ✅ Step 3: Load Model (Use `compile=False` for inference only)
try:
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects, compile=False)
    print("✅ Model Loaded Successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# ✅ Step 4: Define Image Preprocessing Function
def preprocess_image(img_path, image_size=224):
    img = image.load_img(img_path, target_size=(image_size, image_size))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize
    return img_array

# ✅ Step 5: Define Class Labels (Update with your dataset labels)
class_labels = ["Basal Cell Carcinoma", "Benign Keratosis", "Melanoma", "Melanocytic Nevi"]  # Update as needed

# ✅ Step 6: Define Prediction Function
def predict_image(img_path):
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)

    predicted_class_index = np.argmax(predictions)  # Get the highest confidence index
    predicted_class = class_labels[predicted_class_index]  # Get class name
    confidence = np.max(predictions) * 100  # Confidence score in %

    print(f"\n🖼️ Image: {img_path}")
    print(f"🔬 Predicted Disease: {predicted_class} with {confidence:.2f}% confidence\n")
    
    # ✅ Display all class probabilities
    for i, label in enumerate(class_labels):
        print(f"   - {label}: {predictions[0][i] * 100:.2f}%")

# ✅ Step 7: Test with an Image
IMAGE_PATH = "D:/Verdant/SKIN_DESEASES/mel.jpeg"  # Update with your test image path
predict_image(IMAGE_PATH)
