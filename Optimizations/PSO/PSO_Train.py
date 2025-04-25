import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import pyswarms as ps

# تفعيل استخدام GPU بشكل آمن
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("✅ GPU detected and memory growth enabled.")
else:
    print("❌ No GPU found. Please enable it in Colab settings.")

# تحميل بيانات الالتهاب الرئوي
pneumonia_path = "./data_split/"

def load_data():
    train_gen = ImageDataGenerator(rescale=1./255)
    val_gen = ImageDataGenerator(rescale=1./255)

    train_data = train_gen.flow_from_directory(
        os.path.join(pneumonia_path, 'train'),
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary'
    )

    val_data = val_gen.flow_from_directory(
        os.path.join(pneumonia_path, 'val'),
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary'
    )

    return train_data, val_data

train_data, val_data = load_data()

# دالة بناء نموذج VGG16
def build_model(params):
    lr, dropout = params
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
    for layer in base_model.layers:
        layer.trainable = False

    x = Flatten()(base_model.output)
    x = Dense(256, activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=x)
    model.compile(optimizer=Adam(learning_rate=lr), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# دالة اللياقة الخاصة بـ PSO
def pso_fitness(params):
    scores = []
    for p in params:
        try:
            model = build_model(p)
            history = model.fit(train_data, validation_data=val_data, epochs=3, batch_size=32, verbose=0)
            acc = history.history['val_accuracy'][-1]
        except:
            acc = 0  # لو حصل خطأ أو نفاد في الذاكرة
        scores.append(-acc)  # نستخدم السالب لأن PSO يحاول يقلل القيمة
    return np.array(scores)

# إعداد حدود البحث
bounds = (np.array([0.0001, 0.2]), np.array([0.01, 0.5]))

# تهيئة PSO
optimizer = ps.single.GlobalBestPSO(
    n_particles=5,
    dimensions=2,
    options={'c1': 0.5, 'c2': 0.3, 'w': 0.9},
    bounds=bounds
)

# تنفيذ PSO
best_cost, best_pos = optimizer.optimize(pso_fitness, iters=5)

print(f"\n✅ أفضل معلمات: Learning Rate = {best_pos[0]}, Dropout Rate = {best_pos[1]}")

# تدريب النموذج النهائي بالمعلمات المثلى
final_model = build_model(best_pos)
final_model.fit(train_data, validation_data=val_data, epochs=10, batch_size=32)

# حفظ النموذج
final_model.save("best_vgg16_pso_model.keras")
print("✅ النموذج تم حفظه باسم best_vgg16_pso_model.keras")