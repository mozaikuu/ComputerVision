# 📦 Main packages
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tkinter import Tk, filedialog
import torch
from torchvision import transforms
from PIL import Image
from ultralytics import YOLO

# 📤 Choose an image
print("📤 Choose an image..")
Tk().withdraw()
image_path = filedialog.askopenfilename(title="Choose an image", filetypes=[("Image files", "*.jpg *.jpeg *.png")])
if not image_path:
    print("❌ No image chosen")
    exit()

# 🖼️ Try both 150x150 and 180x180
def try_image_load():
    for size in [(150, 150), (180, 180)]:
        try:
            # TensorFlow format
            image_tf = load_img(image_path, target_size=size)
            image_array_tf = img_to_array(image_tf) / 255.0
            image_array_tf = np.expand_dims(image_array_tf, axis=0)

            # PyTorch format
            image_pil = Image.open(image_path).convert("RGB")
            transform_pt = transforms.Compose([
                transforms.Resize(size),
                transforms.ToTensor()
            ])
            image_tensor_pt = transform_pt(image_pil).unsqueeze(0)
            return size, image_array_tf, image_tensor_pt
        except Exception:
            continue
    print("❌ Failed to process the image for both sizes")
    exit()

img_size_used, image_array_tf, image_tensor_pt = try_image_load()

# 📂 Load all models
model_dir = "models"
if not os.path.exists(model_dir):
    print(f"❌ Models folder not found: {model_dir}")
    exit()

results = []

for file in os.listdir(model_dir):
    model_path = os.path.join(model_dir, file)

    try:
        if file.endswith(".keras"):
            print(f"🧠 TF: Loading model {file}")
            model = load_model(model_path)
            prediction = model.predict(image_array_tf)
            confidence = float(prediction[0][0]) * 100
            label = "Pneumonia" if prediction[0][0] > 0.5 else "Normal"
            results.append((file, label, confidence))

        elif file.endswith(".pt"):
            # Try PyTorch model
            try:
                print(f"🧠 Torch: Loading model {file}")
                model = torch.load(model_path, map_location="cpu")
                model.eval()
                with torch.no_grad():
                    output = model(image_tensor_pt)
                    if output.numel() == 1:  # Binary
                        prob = torch.sigmoid(output).item()
                    else:  # Multi-class
                        prob = torch.softmax(output, dim=1)[0][1].item()
                    confidence = prob * 100
                    label = "Pneumonia" if prob > 0.5 else "Normal"
                    results.append((file, label, confidence))

            except Exception as e2:
                # Try YOLOv8 fallback
                try:
                    print(f"🧠 YOLO: Loading model {file}")
                    model = YOLO(model_path)
                    yolo_result = model.predict(image_path)[0]

                    # Detect classification or detection
                    if yolo_result.probs is not None:  # Classification model
                        label_index = int(torch.argmax(yolo_result.probs))
                        confidence = float(yolo_result.probs[label_index]) * 100
                        label = model.names[label_index]
                    else:  # Object detection
                        if len(yolo_result.boxes.conf) > 0:
                            confidence = float(yolo_result.boxes.conf[0]) * 100
                            label = model.names[int(yolo_result.boxes.cls[0])]
                        else:
                            confidence = 0
                            label = "No object detected"

                    results.append((file, label, confidence))

                except Exception as e3:
                    print(f"⚠️ Error with YOLO model {file}: {e3}")

    except Exception as e:
        print(f"⚠️ Error with model {file}: {e}")

# 📊 Display results
img_display = cv2.imread(image_path)
img_display = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 8))
plt.imshow(img_display)
plt.axis("off")
title_lines = [f"{file}: {label} ({conf:.2f}%)" for file, label, conf in results]
plt.title("\n".join(title_lines), fontsize=10)
plt.tight_layout()
plt.show()
