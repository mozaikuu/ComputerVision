# 📦 استيراد المكتبات الأساسية
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
# from google.colab import files

# ✅ تحميل النموذج المدرب
final_model = load_model("best_vgg16_pso_model.keras")

# 📁 إنشاء مجلد لحفظ الصور إذا لم يكن موجودًا
image_folder = "test_images"
os.makedirs(image_folder, exist_ok=True)

# ⬆ رفع صورة من الجهاز المحلي
print("📤 Choose a picture to upload")
uploaded = files.upload()

# 💾 حفظ الصور المرفوعة في مجلد test_images
for filename in uploaded.keys():
    file_content = uploaded[filename]
    full_path = os.path.join(image_folder, filename)
    with open(full_path, 'wb') as f:
        f.write(file_content)
    print(f"✅ Saved: {filename} في المجلد {image_folder}")

# 📂 عرض الصور المتاحة
image_files = os.listdir(image_folder)
print("📂 Available images:", image_files)

# 🖼 إدخال اسم الصورة للتصنيف
image_name = input("📸 Enter image name from the list above: ")
image_path = os.path.join(image_folder, image_name)

# ✅ التأكد من أن الصورة موجودة
if not os.path.exists(image_path):
    print("❌ Image not found! Please check the name.")
else:
    try:
        # 🧼 معالجة الصورة (نفس الحجم اللي تدرب عليه النموذج)
        img_size = (150, 150)
        image = load_img(image_path, target_size=img_size)
        image_array = img_to_array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        # 🔍 التنبؤ باستخدام النموذج
        prediction = final_model.predict(image_array)
        confidence = float(prediction[0][0]) * 100
        label = "Pneumonia" if prediction[0][0] > 0.5 else "Normal"

        # 🖼 عرض الصورة والنتيجة
        img_display = cv2.imread(image_path)
        img_display = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)

        plt.imshow(img_display)
        plt.axis("off")
        plt.title(f"🔍 Diagnosis: {label} ({confidence:.2f}%)", fontsize=14)
        plt.show()

    except Exception as e:
        print(f"⚠ Error while processing the image: {e}")

# 💾 تحميل النموذج
files.download("best_vgg16_pso_model.keras")