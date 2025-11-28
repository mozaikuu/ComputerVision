# 📦 main packages
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(BASE_DIR, "models")
test_dir = os.path.join(BASE_DIR, "test_images")

# 🧪 test directory structure (e.g., test/Normal/, test/Pneumonia/)
img_size = (224, 224)
class_names = ["Normal", "Pneumonia"]

def load_test_data():
    X = []
    y = []
    for label_index, class_name in enumerate(class_names):
        class_path = os.path.join(test_dir, class_name)
        for img_file in os.listdir(class_path):
            img_path = os.path.join(class_path, img_file)
            try:
                image = load_img(img_path, target_size=img_size)
                image_array = img_to_array(image) / 255.0
                X.append(image_array)
                y.append(label_index)
            except Exception as e:
                print(f"⚠️ Skipping {img_path}: {e}")
    return np.array(X), np.array(y)

# 🧠 Evaluate Keras models

if not os.path.exists(model_dir):
    print(f"❌ Models folder not found: {model_dir}")
    exit()

X_test, y_test = load_test_data()
print(f"✅ Loaded {len(X_test)} test images")

for file in os.listdir(model_dir):
    if file.endswith(".keras"):
        model_path = os.path.join(model_dir, file)
        print(f"\n🧠 Evaluating model: {file}")
        try:
            model = load_model(model_path)
            y_pred_probs = model.predict(X_test)
            y_pred = (y_pred_probs > 0.5).astype("int32").flatten()

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            print(f"📊 Results for {file}:")
            print(f"   Accuracy : {acc * 100:.2f}%")
            print(f"   Precision: {prec:.2f}")
            print(f"   Recall   : {rec:.2f}")
            print(f"   F1 Score : {f1:.2f}")
        except Exception as e:
            print(f"⚠️ Error evaluating {file}: {e}")
