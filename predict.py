import tensorflow as tf
import numpy as np
from PIL import Image

# Model and image settings
MODEL_PATH = "best_solar_panel_model.keras"
IMG_SIZE = (224, 224)

# Class names
class_names = [
    "Bird-drop",
    "Clean",
    "Dusty",
    "Electrical-damage",
    "Physical-Damage",
    "Snow-Covered"
]

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Ask for image path
image_path = input("Enter the image path: ")

# Load and prepare image
image = Image.open(image_path).convert("RGB")
image = image.resize(IMG_SIZE)

image_array = np.array(image)
image_array = np.expand_dims(image_array, axis=0)

# Predict
predictions = model.predict(image_array, verbose=0)

predicted_index = np.argmax(predictions[0])
predicted_class = class_names[predicted_index]
confidence = predictions[0][predicted_index] * 100

# Display result
print("\n==============================")
print("Solar Panel Fault Detection")
print("==============================")
print("Prediction:", predicted_class)
print(f"Confidence: {confidence:.2f}%")
print("==============================")