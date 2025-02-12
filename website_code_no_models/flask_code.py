from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import random
from tensorflow.keras.applications.resnet50 import preprocess_input


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# INSERT TRAINED MODEL HERE
model = tf.keras.models.load_model("test_model_2.h5")

# test_model_2 tended to perform best out of all models we tried

# MobileNet model - we tested some MobileNet codes and this was a framework for running those models
# def preprocess_image(img_path):
#     try:
#         # Resize the image to (256, 256) as expected by MobileNet
#         img = image.load_img(img_path, target_size=(256, 256))
#         img_array = image.img_to_array(img) / 255.0  # Normalize pixel values to [0, 1]
#         img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

#         # Apply MobileNet-specific preprocessing (e.g., scale to [-1, 1])
#         img_array = preprocess_input(img_array)  # This adjusts the pixel range as required for MobileNet

#         return img_array
#     except Exception as e:
#         print(f"Error loading image: {e}")
#         raise


# For ResNet Models
def preprocess_image(img_path):
    try:
        # Resize the image to (224, 224) as expected by ResNet50
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)  # Convert to NumPy array
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Apply ResNet50-specific preprocessing (DO NOT divide by 255)
        img_array = preprocess_input(img_array)  

        return img_array
    except Exception as e:
        print(f"Error loading image: {e}")
        

@app.route('/')
def home():
    return render_template("index.html")

# Return breed prediction with confidence 
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(img_path)

    # Preprocess the image
    img_array = preprocess_image(img_path)

    # Get prediction probabilities
    predictions = model.predict(img_array)  # This will be an array of probabilities
    predicted_class = np.argmax(predictions)  # Index of highest probability
    predicted_breed = class_labels[predicted_class]
    confidence_score = float(np.max(predictions))  # Highest probability

    return jsonify({"breed": predicted_breed, "confidence": confidence_score})

# Test images are stored in 'test_images/' and class_labels is a list of breed names
TEST_IMAGES_DIR = "static/test_images"
LABELS_PATH = os.path.join(os.path.dirname(__file__), "static", "class_labels.txt")

# Load class labels from the file
with open(LABELS_PATH, "r") as f:
    class_labels = [line.strip() for line in f]

# Load class labels from the file
with open(LABELS_PATH, "r") as f:
    class_labels = [line.strip() for line in f]

@app.route("/random_image")
def random_image():
    test_images = os.listdir(TEST_IMAGES_DIR)
    random_image = random.choice(test_images)
    random_breed = random_image.split("_")[0]  # Assuming filenames are like "Labrador_1.jpg"

    print(f"Selected Image: {random_image}, Breed: {random_breed}")  # Debugging

    return jsonify({"image_url": f"/static/test_images/{random_image}", "breed": random_breed})

if __name__ == "__main__":
    app.run(debug=True)

