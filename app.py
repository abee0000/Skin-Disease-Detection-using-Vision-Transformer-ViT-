import tensorflow as tf
from transformers import TFViTForImageClassification
from PIL import Image, ImageTk
import numpy as np
import tkinter as tk
from tkinter import filedialog, StringVar, OptionMenu
from tkinterdnd2 import DND_FILES, TkinterDnD

# --- CONFIG ---
model_path = "final_model_001"
img_size = (224, 224)
class_names = {0: 'Basal Cell Carcinoma', 1: 'Benign Keratosis Like', 2: 'Melanoma', 3: 'Melanocytic Nevi'}

# --- LOAD MODEL ---
print("🔄 Loading model...")
model = TFViTForImageClassification.from_pretrained(model_path)
print("✅ Model loaded.")

# --- IMAGE PROCESSING ---
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB").resize(img_size)
    img_array = np.array(image).astype(np.float32) / 255.0
    img_array = (img_array - 0.5) / 0.5
    img_array = np.transpose(img_array, (2, 0, 1))
    img_array = np.expand_dims(img_array, axis=0)
    return tf.convert_to_tensor(img_array), image

def predict(image_path):
    image_tensor, pil_img = preprocess_image(image_path)
    logits = model(image_tensor, training=False).logits
    probs = tf.nn.softmax(logits, axis=-1).numpy()[0]
    pred_index = np.argmax(probs)
    top3_indices = probs.argsort()[-3:][::-1]
    top3 = [(class_names[i], probs[i]) for i in top3_indices]
    return class_names[pred_index], probs[pred_index], top3, pil_img

# --- GUI LOGIC ---
def show_prediction(image_path):
    label_file.config(text=image_path)
    pred_label, pred_conf, top3, img = predict(image_path)

    # Show Image
    img_resized = img.resize((250, 250))
    tk_img = ImageTk.PhotoImage(img_resized)
    image_label.config(image=tk_img)
    image_label.image = tk_img

    # Display Main Result
    result_label.config(text=f"✅ Prediction: {pred_label}\n📊 Confidence: {pred_conf * 100:.2f}%")

    # Update Dropdown for Top-3
    top3_options = [f"{label}: {conf * 100:.2f}%" for label, conf in top3]
    selected.set("Top-3 Predictions")
    dropdown_menu['menu'].delete(0, 'end')
    for option in top3_options:
        dropdown_menu['menu'].add_command(label=option, command=tk._setit(selected, option))

def on_drop(event):
    filepath = event.data.strip('{}')
    show_prediction(filepath)

def browse_file():
    filepath = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
    if filepath:
        show_prediction(filepath)

# --- STYLISH UI ---
root = TkinterDnD.Tk()
root.title("🧠 Skin Disease Classifier")
root.geometry("600x700")
root.configure(bg="#f7faff")

title = tk.Label(root, text="🧪 AI Skin Disease Classifier", font=("Helvetica", 18, "bold"), bg="#f7faff", fg="#1e3d59")
title.pack(pady=15)

label_file = tk.Label(root, text="📁 Drop an image or click Browse", font=("Helvetica", 12), bg="#f7faff", fg="#3b3b3b")
label_file.pack(pady=5)

drop_frame = tk.Label(root, width=55, height=6, bg="#e3f2fd", fg="#0d47a1", text="⬇️ Drag & Drop Image Here",
                      font=("Helvetica", 12, "bold"), relief="solid", borderwidth=2)
drop_frame.pack(pady=10)
drop_frame.drop_target_register(DND_FILES)
drop_frame.dnd_bind('<<Drop>>', on_drop)

btn_browse = tk.Button(root, text="📂 Browse Image", command=browse_file,
                       font=("Helvetica", 11, "bold"), bg="#1976d2", fg="white",
                       padx=10, pady=5, relief="raised", bd=3)
btn_browse.pack(pady=10)

image_label = tk.Label(root, bg="#f7faff")
image_label.pack(pady=10)

# Main Result
result_label = tk.Label(root, text="", bg="#f7faff", fg="#333333",
                        font=("Helvetica", 12), justify="center")
result_label.pack(pady=10)

# Dropdown for Top-3
selected = StringVar()
selected.set("Top-3 Predictions")
dropdown_menu = OptionMenu(root, selected, "Top-3 Predictions")
dropdown_menu.config(font=("Helvetica", 11), bg="white", fg="#0d47a1", width=30)
dropdown_menu.pack(pady=5)

footer = tk.Label(root, text="🔬 Powered by Vision Transformer", font=("Helvetica", 9, "italic"), bg="#f7faff", fg="#888888")
footer.pack(side="bottom", pady=10)

root.mainloop()
