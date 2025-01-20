from flask import Flask, request, render_template
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import numpy as np
from io import BytesIO

# Load the trained model
model = tf.keras.models.load_model('cnn_image_classifier.h5')

# Define class names for the CIFAR-10 dataset
class_names = ["Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]

# Initialize the Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')  # Ensure you have an index.html file in the templates folder

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file uploaded!', 400

    file = request.files['file']
    if file.filename == '':
        return 'No selected file!', 400

    try:
        # Preprocess the image
        img = image.load_img(BytesIO(file.read()), target_size=(32, 32))   # Use file.stream here
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict the class
        predictions = model.predict(img_array)
        predicted_label = np.argmax(predictions[0])
        result = class_names[predicted_label]

        return render_template('result.html', result=result)  # Pass result to the result.html template
    except Exception as e:
        return f"Error processing the image: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
