# Define the data directory path for local use
import os
import shutil
import random
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Set the base data directory
BASE_DIR = 'd:/2_University/Projects/Sandy/ComputerVision/data'

# Define image size for preprocessing
IMG_SIZE = 224  # Standard size for many image classification models

# Define directories for train, test, val splits
SPLIT_DIR = os.path.join(os.path.dirname(BASE_DIR), 'data_split')
os.makedirs(SPLIT_DIR, exist_ok=True)

TRAIN_DIR = os.path.join(SPLIT_DIR, 'train')
TEST_DIR = os.path.join(SPLIT_DIR, 'test')
VAL_DIR = os.path.join(SPLIT_DIR, 'val')

# Create directories if they don't exist
for dir_path in [TRAIN_DIR, TEST_DIR, VAL_DIR]:
    os.makedirs(dir_path, exist_ok=True)
    # Create class subdirectories
    for class_name in ['NORMAL', 'PNEUMONIA']:
        os.makedirs(os.path.join(dir_path, class_name), exist_ok=True)

# Function to preprocess and resize images
def preprocess_image(img_path, output_path, img_size=IMG_SIZE):
    try:
        # Read image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to read image: {img_path}")
            return False
        
        # Resize image to the same dimensions
        resized_img = cv2.resize(img, (img_size, img_size))
        
        # Save the preprocessed image
        cv2.imwrite(output_path, resized_img)
        return True
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return False

# Function to split and preprocess the dataset
def split_and_preprocess_dataset():
    # Check if split already exists
    if os.path.exists(os.path.join(TRAIN_DIR, 'NORMAL')) and \
       len(os.listdir(os.path.join(TRAIN_DIR, 'NORMAL'))) > 0:
        print("Dataset already split and preprocessed.")
        return
    
    # Get all image paths for each class
    dataset = {}
    for class_name in ['NORMAL', 'PNEUMONIA']:
        class_dir = os.path.join(BASE_DIR, class_name)
        if not os.path.exists(class_dir):
            print(f"Class directory not found: {class_dir}")
            continue
            
        image_paths = [os.path.join(class_dir, img_name) for img_name in os.listdir(class_dir) 
                      if img_name.lower().endswith(('.png', '.jpg', '.jpeg'))]
        dataset[class_name] = image_paths
        print(f"Found {len(image_paths)} images for class {class_name}")
    
    # Split the dataset for each class
    for class_name, image_paths in dataset.items():
        # First split: 80% train, 20% temp (for test and validation)
        train_paths, temp_paths = train_test_split(image_paths, test_size=0.2, random_state=42)
        
        # Second split: 50% test, 50% validation (from the temp set)
        test_paths, val_paths = train_test_split(temp_paths, test_size=0.5, random_state=42)
        
        print(f"Class {class_name} - Train: {len(train_paths)}, Test: {len(test_paths)}, Val: {len(val_paths)}")
        
        # Process and copy images to their respective directories
        for img_path, target_dir in [
            (train_paths, TRAIN_DIR),
            (test_paths, TEST_DIR),
            (val_paths, VAL_DIR)
        ]:
            for path in img_path:
                filename = os.path.basename(path)
                output_path = os.path.join(target_dir, class_name, filename)
                preprocess_image(path, output_path)
    
    print("Dataset split and preprocessing completed.")

# Run the split and preprocessing
split_and_preprocess_dataset()

# Print the dataset structure
for split_name, split_dir in [("Training", TRAIN_DIR), ("Testing", TEST_DIR), ("Validation", VAL_DIR)]:
    print(f"\n{split_name} dataset:")
    for class_name in ['NORMAL', 'PNEUMONIA']:
        class_dir = os.path.join(split_dir, class_name)
        if os.path.exists(class_dir):
            print(f"  {class_name}: {len(os.listdir(class_dir))} images")

# Load a model
model = YOLO("yolov8n-cls.pt")  # load a pretrained model

# Use the model with the preprocessed training data
results = model.train(data=SPLIT_DIR, epochs=5, imgsz=IMG_SIZE)  # train the model

# Display training results
# Check if the results directory exists
results_dir = os.path.join(os.getcwd(), 'runs')
if os.path.exists(results_dir):
    print(f"Results saved to: {results_dir}")
    # You can add code here to visualize results
else:
    print("No results directory found")

# Function to test the model on a custom image
def test_model_on_image(model, image_path):
    # Preprocess the image
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Failed to read image: {image_path}")
            return None
        
        # Resize image to the same dimensions expected by the model
        resized_img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        
        # Run inference
        results = model.predict(source=resized_img, verbose=False)
        
        # Get the prediction results
        result = results[0]
        
        # Display the image with prediction
        plt.figure(figsize=(8, 8))
        plt.imshow(cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        # Get top prediction and confidence
        probs = result.probs.data.tolist()
        class_names = result.names
        top_i = probs.index(max(probs))
        top_class = class_names[top_i]
        top_prob = probs[top_i] * 100  # Convert to percentage
        
        plt.title(f"Prediction: {top_class}\nConfidence: {top_prob:.2f}%")
        plt.show()
        
        # Print detailed results
        print(f"\nPrediction Results:")
        for i, prob in enumerate(probs):
            print(f"{class_names[i]}: {prob*100:.2f}%")
        
        return result
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

# Function to open file dialog and select an image
def select_and_test_image():
    import tkinter as tk
    from tkinter import filedialog
    
    # Create a root window but hide it
    root = tk.Tk()
    root.withdraw()
    
    # Open file dialog to select an image
    file_path = filedialog.askopenfilename(
        title="Select an image to test",
        filetypes=[("Image files", "*.jpg;*.jpeg;*.png")]
    )
    
    if file_path:
        print(f"Selected image: {file_path}")
        # Test the model on the selected image
        test_model_on_image(model, file_path)
    else:
        print("No image selected.")

# Allow user to test the model with a custom image
print("\nWould you like to test the model with a custom image? (y/n)")
user_input = input().strip().lower()
if user_input == 'y':
    select_and_test_image()