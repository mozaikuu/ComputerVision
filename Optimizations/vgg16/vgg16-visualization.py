import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("best_vgg16_model.keras")

# Optional: Replace this with your actual class names
class_names = ["Pneumonia", "Normal"]  # Adjust this if you have more classes

# Function to predict image class
def predict_image(img_path):
    image = tf.keras.preprocessing.image.load_img(img_path, target_size=(150, 150))
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = image_array / 255.0  # Normalize if needed

    predictions = model.predict(image_array)[0]
    predicted_class = np.argmax(predictions)
    confidence = predictions[predicted_class] * 100

    return predicted_class, confidence, predictions

# CustomTkinter App
class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("VGG16 Image Classifier")
        self.geometry("600x500")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # Widgets
        self.label = ctk.CTkLabel(self, text="Upload an Image", font=("Arial", 20))
        self.label.pack(pady=20)

        self.button = ctk.CTkButton(self, text="Choose Image", command=self.choose_image)
        self.button.pack(pady=10)

        self.image_label = ctk.CTkLabel(self, text="")
        self.image_label.pack(pady=10)

        self.result_label = ctk.CTkLabel(self, text="", font=("Arial", 16))
        self.result_label.pack(pady=10)

    def choose_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_path:
            # Display the image
            img = Image.open(file_path)
            img.thumbnail((200, 200))
            img_tk = ImageTk.PhotoImage(img)
            self.image_label.configure(image=img_tk, text="")
            self.image_label.image = img_tk

            # Predict
            predicted_class, confidence, _ = predict_image(file_path)
            class_name = class_names[predicted_class]
            if confidence >= 0.5:
              self.result_label.configure(
                  text=f"Predicted Class: Pneumonia \nConfidence: {confidence:.2f}%"
              )
            else:
              self.result_label.configure(
                  text=f"Predicted Class: Normal \nConfidence: {1 - confidence:.2f}%"
              )

# Run the app
if __name__ == "__main__":
    app = App()
    app.mainloop()
