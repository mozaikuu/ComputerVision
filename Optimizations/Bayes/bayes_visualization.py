# 📦 استيراد المكتبات الأساسية
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from google.colab import files

# ✅ تحميل النموذج المدرب
final_model = load_model("best_vgg16_bayes_model.keras")

# 📁 إنشاء مجلد لحفظ الصور إذا لم يكن موجودًا
image_folder = "test_images"
os.makedirs(image_folder, exist_ok=True)

# ⬆ رفع صورة/صور من الجهاز المحلي
print("📤 اختر صورة أو أكثر للرفع")
uploaded = files.upload()

# 💾 حفظ الصور في مجلد test_images
for filename in uploaded.keys():
    file_content = uploaded[filename]
    full_path = os.path.join(image_folder, filename)
    with open(full_path, 'wb') as f:
        f.write(file_content)
    print(f"✅ تم حفظ الصورة: {filename}")

# 📂 الحصول على جميع الصور المرفوعة
image_files = os.listdir(image_folder)
print("📂 الصور التي سيتم تصنيفها:", image_files)

# 🖼 تصنيف كل صورة وعرض النتيجة
for image_name in image_files:
    image_path = os.path.join(image_folder, image_name)
    try:
        # 🧼 تجهيز الصورة
        img_size = (150, 150)
        image = load_img(image_path, target_size=img_size)
        image_array = img_to_array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        # 🔍 التنبؤ
        prediction = final_model.predict(image_array)
        confidence = float(prediction[0][0]) * 100
        label = "Pneumonia" if prediction[0][0] > 0.5 else "Normal"

        # 🖼 عرض الصورة والنتيجة
        img_display = cv2.imread(image_path)
        img_display = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)

        plt.imshow(img_display)
        plt.axis("off")
        plt.title(f"{image_name}\n🔍 Diagnosis: {label} ({confidence:.2f}%)", fontsize=14)
        plt.show()

    except Exception as e:
        print(f"⚠ حصل خطأ في الصورة {image_name}: {e}")

# 💾 تحميل النموذج لو محتاجة تحفظيه
files.download("best_vgg16_bayes_model.keras")