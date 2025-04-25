import tensorflow as tf

# ✅ ضبط GPU قبل أي استخدام لـ TensorFlow
# تأكد من تفعيل استخدام الذاكرة بشكل آمن قبل أي شيء آخر
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ {len(gpus)} GPU(s) detected. Memory growth enabled.")
    except RuntimeError as e:
        print(f"⚠️ Failed to set memory growth: {e}")
else:
    print("❌ No GPU found. Please check your TensorFlow installation and GPU drivers.")

import os
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from skopt import gp_minimize
from skopt.space import Real
# import kagglehub


# تحميل بيانات الالتهاب الرئوي
# pneumonia_path = kagglehub.dataset_download('paultimothymooney/chest-xray-pneumonia')
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

# بناء نموذج VGG16
def build_model(lr, dropout):
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

# دالة الهدف لـ Bayesian Optimization
def fitness(params):
    lr, dropout = params
    try:
        model = build_model(lr, dropout)
        history = model.fit(train_data, validation_data=val_data, epochs=3, batch_size=32, verbose=0)
        val_acc = history.history['val_accuracy'][-1]
        print(f"lr: {lr:.5f}, dropout: {dropout:.2f}, acc: {val_acc:.4f}")
        return -val_acc  # نريد تعظيم الدقة، فنعكس القيمة
    except Exception as e:
        print("⚠ Error:", e)
        return 1.0  # قيمة سيئة في حال فشل التدريب

# البحث باستخدام Bayesian Optimization
search_space = [Real(1e-4, 1e-2, name='learning_rate'), Real(0.2, 0.5, name='dropout')]

result = gp_minimize(
    func=fitness,
    dimensions=search_space,
    acq_func='EI',  # Expected Improvement
    n_calls=10,
    random_state=42
)

best_lr, best_dropout = result.x
print(f"\n✅ أفضل معلمات: Learning Rate = {best_lr}, Dropout = {best_dropout}")

# تدريب النموذج النهائي
final_model = build_model(best_lr, best_dropout)
final_model.fit(train_data, validation_data=val_data, epochs=10, batch_size=32)

# حفظ النموذج
final_model.save("best_vgg16_bayes_model.keras")
print("✅ النموذج تم حفظه باسم best_vgg16_bayes_model.keras")