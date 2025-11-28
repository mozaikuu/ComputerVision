# 📦 main packages
import os
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

# 📂 Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Ensure the script looks for 'models' and 'test_images' in its own directory
model_dir = os.path.join(BASE_DIR, "models")
test_dir = os.path.join(BASE_DIR, "test_images")

# ⚙️ Image settings
img_size = (150, 150)
class_names = ["Normal", "Pneumonia"]

# 🔄 Load test data
def load_test_data():
    """Loads images and labels from the test directory."""
    X = []
    y = []
    print("--- Loading Test Data ---")
    for label_index, class_name in enumerate(class_names):
        class_path = os.path.join(test_dir, class_name)
        if not os.path.exists(class_path):
            print(f"⚠️ Class folder missing: {class_path}")
            continue
        image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
        for img_file in image_files:
            img_path = os.path.join(class_path, img_file)
            try:
                image = load_img(img_path, target_size=img_size)
                image_array = img_to_array(image) / 255.0
                X.append(image_array)
                y.append(label_index)
            except Exception as e:
                print(f"⚠️ Skipping {img_path}: {e}")
    if not X:
        print("❌ No test images were loaded. Please check the 'test_images' directory structure.")
        exit()
    return np.array(X), np.array(y)

# 🧪 Evaluate models
if not os.path.exists(model_dir):
    print(f"❌ Models folder not found: {model_dir}")
    exit()

X_test, y_test = load_test_data()
print(f"✅ Loaded {len(X_test)} test images.")

results = []
model_files = [f for f in os.listdir(model_dir) if f.endswith(".keras")]

if not model_files:
    print(f"❌ No '.keras' models found in the '{model_dir}' folder.")
    exit()

for file in model_files:
    model_path = os.path.join(model_dir, file)
    model_name = file.replace(".keras", "")
    print(f"\n🧠 Evaluating model: {model_name}")
    try:
        model = load_model(model_path)

        # --- Efficiency Metrics ---
        start_time = time.time()
        y_pred_probs = model.predict(X_test)
        end_time = time.time()
        total_time = end_time - start_time
        avg_inference_time_ms = (total_time / len(X_test)) * 1000
        model_size_mb = os.path.getsize(model_path) / (1024 * 1024)

        # --- Performance Metrics ---
        y_pred = (y_pred_probs > 0.5).astype("int32").flatten()
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_pred_probs)

        results.append({
            "Model": model_name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1_Score": f1,
            "AUC": auc,
            "Inference_Time_ms": avg_inference_time_ms,
            "Size_MB": model_size_mb
        })

        print(f"📊 Results for {model_name}:")
        print(f"   Accuracy : {acc * 100:.2f}% | AUC: {auc:.4f} | F1 Score: {f1:.4f}")
        print(f"   Inference: {avg_inference_time_ms:.4f} ms/image | Size: {model_size_mb:.2f} MB")
        
        # --- INDIVIDUAL PLOTS ---
        # 1. Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix: {model_name}")
        plt.show()

        # 2. ROC Curve
        RocCurveDisplay.from_predictions(y_test, y_pred_probs, name=model_name)
        plt.title(f"ROC Curve: {model_name}")
        plt.plot([0, 1], [0, 1], "k--", label="Chance Level (AUC = 0.5)")
        plt.legend()
        plt.show()

    except Exception as e:
        print(f"⚠️ Error evaluating {file}: {e}")

# 📊 Convert to DataFrame and analyze
if results:
    df_results = pd.DataFrame(results)
    print("\n\n✅ All Combined Results:")
    print(df_results)
    df_results.to_csv("model_results_detailed.csv", index=False)
    print("\n✅ Detailed results saved to 'model_results_detailed.csv'")

    # --- Variance Metrics ---
    print("\n--- Variance of Performance Metrics Across Models ---")
    performance_metrics = ["Accuracy", "Precision", "Recall", "F1_Score", "AUC"]
    performance_variance = df_results[performance_metrics].var()
    print(performance_variance)

    # ==============================================
    # ✅ PLOT 1: Combined Barplot with Labels
    # ==============================================
    sns.set(style="whitegrid")
    melted = df_results.melt(
        id_vars="Model",
        value_vars=performance_metrics,
        var_name="Metric",
        value_name="Score"
    )
    plt.figure(figsize=(12, 7))
    ax = sns.barplot(data=melted, x="Model", y="Score", hue="Metric")
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f"{height:.2%}", (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom', fontsize=9, color="black",
                    xytext=(0, 3), textcoords='offset points')
    plt.title("Model Performance Comparison")
    plt.ylim(max(0, melted["Score"].min() - 0.05), 1.0)
    plt.ylabel("Score")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    # ==============================================
    # ✅ PLOT 2: Heatmap
    # ==============================================
    heatmap_data = df_results.set_index("Model")[performance_metrics]
    plt.figure(figsize=(10, 5))
    sns.heatmap(heatmap_data * 100, annot=True, fmt=".2f", cmap="viridis", linewidths=0.5)
    plt.title("Model Metrics Heatmap (%)")
    plt.tight_layout()
    plt.show()

    # ==============================================
    # ✅ PLOT 3: Radar (Spider) Plot
    # ==============================================
    labels = performance_metrics
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist() + [0]
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for i, row in df_results.iterrows():
        values = row[labels].tolist() + [row[labels][0]]
        ax.plot(angles, values, label=row["Model"], linewidth=2)
        ax.fill(angles, values, alpha=0.1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    plt.title("Radar Plot of Model Metrics", size=15, y=1.1)
    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.show()
    
    # ==============================================
    # ✅ PLOT 4: NEW - Efficiency Plot
    # ==============================================
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=df_results,
        x="Size_MB",
        y="Inference_Time_ms",
        hue="Model",
        size="AUC",
        sizes=(100, 1000),
        palette="viridis",
        alpha=0.8
    )
    plt.title("Model Efficiency: Inference Time vs. Size (Bubble size by AUC)")
    plt.xlabel("Model Size (MB)")
    plt.ylabel("Average Inference Time per Image (ms)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ==============================================
    # ✅ PLOT 5: Sorted Barplots Per Metric
    # ==============================================
    for metric in performance_metrics:
        plt.figure(figsize=(7, 5))
        sorted_df = df_results.sort_values(metric, ascending=False)
        ax = sns.barplot(x=metric, y="Model", data=sorted_df, palette="Blues_d")
        ax.bar_label(ax.containers[0], fmt='{:.2%}', padding=5)
        plt.title(f"{metric} by Model (Sorted)")
        plt.xlim(0, 1.05)
        plt.tight_layout()
        plt.show()
        
else:
    print("⚠️ No results were generated to display.")


# ==============================================
# ✅ EXPLANATION FOR CONVERGENCE PLOTS
# ==============================================
print("""
---------------------------------------------------------------------
💡 A Note on Convergence (Training History) Plots:
---------------------------------------------------------------------
Convergence plots (like loss vs. epochs) cannot be generated from saved
'.keras' files because the training history is not stored in the model file.

To get these plots, you must save the 'history' object that is returned
by the `model.fit()` command during the original training process.

Example of how to do it during training:
-----------------------------------------

# During training:
# history = model.fit(
#     train_generator,
#     epochs=50,
#     validation_data=validation_generator
# )

# After training, you would use the 'history' object to plot:
# pd.DataFrame(history.history).plot(figsize=(10, 6))
# plt.grid(True)
# plt.title("Model Training & Validation History")
# plt.xlabel("Epoch")
# plt.gca().set_ylim(0, 1) # Optional: set the y-axis range
# plt.show()
---------------------------------------------------------------------
""")