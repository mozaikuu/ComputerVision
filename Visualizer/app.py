# 📦 main packages
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
import shutil
from ultralytics import YOLO

# 🖼️ img choice
print("📤 Choose an image..")
Tk().withdraw()
image_path = filedialog.askopenfilename(title="Choose an image", filetypes=[("Image files", "*.jpg *.jpeg *.png")])
if not image_path:
    print("❌ no image chosen")
    exit()

def img_morphing():
    try:
        img_size = (150, 150)
        image = load_img(image_path, target_size=img_size)
        image_array_tf = img_to_array(image) / 255.0
        image_array_tf = np.expand_dims(image_array_tf, axis=0)
        
        # تحويل لـ PIL لـ PyTorch
        image_pil = Image.open(image_path).convert("RGB")
        transform_pt = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor()
        ])
        image_tensor_pt = transform_pt(image_pil).unsqueeze(0)
        
    except Exception as e1:
        print("⚠️ Failed with size (150, 150), trying (180, 180)...")
        
        try:
            img_size = (180, 180)
            image = load_img(image_path, target_size=img_size)
            image_array_tf = img_to_array(image) / 255.0
            image_array_tf = np.expand_dims(image_array_tf, axis=0)
            
            # تحويل لـ PIL لـ PyTorch
            image_pil = Image.open(image_path).convert("RGB")
            transform_pt = transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor()
            ])
            image_tensor_pt = transform_pt(image_pil).unsqueeze(0)
                    
        except Exception as e2:
            print(f"❌ Failed to load image with both sizes: {e2}")
            exit()
    
    return image_array_tf, image_tensor_pt

# 📂 البحث عن جميع النماذج
model_dir = "models"
if not os.path.exists(model_dir):
    print(f"❌ Models folder not found: {model_dir}")
    exit()

results = []

for file in os.listdir(model_dir):
    model_path = os.path.join(model_dir, file)

    try:
        # Checking for Keras models
        if file.endswith(".keras"):
            print(f"🧠 TF: Loading model {file}")
            model = load_model(model_path)
            prediction = model.predict(img_morphing()[0])
            confidence = float(prediction[0][0]) * 100
            label = "Pneumonia" if prediction[0][0] > 0.5 else "Normal"
            results.append((file, label, confidence))

        # # Checking for PyTorch models
        elif file.endswith(".pt"):
            try:
                print(f"🧠 Torch: Loading model {file}")
                model = torch.load(model_path, map_location=torch.device("cpu"), weights_only=False)
                with torch.no_grad():                
                    output = model(img_morphing()[1])
                    prob = torch.sigmoid(output).item() if output.numel() == 1 else torch.softmax(output, dim=1)[0][1].item()
                    confidence = prob * 100
                    label = "Pneumonia" if prob > 0.5 else "Normal"
                    results.append((file, label, confidence))
                    model.eval()
            except Exception as e2:
                print(f"⚠️ Failed with Torch model {file}: {e2}")

                # Checking for YOLO models if Torch fails
                try:
                    print(f"🧠 YOLO: Loading model {file}")
                    model = YOLO(model_path)  # Correct way to load YOLO models
                    result = model.predict(image_path)  # Predict directly from image path
                    confidence = result.pandas().xywh['confidence'][0] * 100  # Get confidence score
                    label = result.pandas().xywh['class'][0]  # Label is the class of the detected object
                    results.append((file, label, confidence))
                except Exception as e3:
                    print(f"⚠️ Error with YOLO model {file}: {e3}")
                
    except Exception as e:
        print(f"⚠️ Error with model {file}: {e}")

# 🖼️ عرض النتائج على الصورة
img_display = cv2.imread(image_path)
img_display = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(8, 6))
plt.imshow(img_display)
plt.axis("off")

title_lines = [f"{file}: {label} ({conf:.2f}%)" for file, label, conf in results]
plt.title("\n".join(title_lines), fontsize=12)
plt.tight_layout()
plt.show()
