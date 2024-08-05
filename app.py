from flask import Flask, request, jsonify
import numpy as np
import cv2
import tflite_runtime.interpreter as tflite
import base64

app = Flask(__name__)

# Load TFLite model and allocate tensors
interpreter = tflite.Interpreter(model_path="model/pet_models.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Class names (replace with your actual class names)
class_names = ['cat', 'dog', 'rabbit']  # Example class names

def preprocess_image(image):
    # Resize the image to match the input shape of your model
    input_shape = input_details[0]['shape'][1:3]  # Assuming NHWC format
    image = cv2.resize(image, input_shape)
    # Normalize the image
    image = image.astype(np.float32) / 255.0
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    return image

@app.route("/")
def index():
    return "Hello from Docker! Use /api/likers, /api/post, or /api/comments with ?url= parameter to get Instagram data."

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' in request.files:
        file = request.files['file']
        image_array = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    elif 'image' in request.json:
        base64_string = request.json['image']['base64']
        image_data = base64.b64decode(base64_string)
        image_array = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    else:
        return jsonify({'error': 'No image provided'}), 400

    processed_image = preprocess_image(image)

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], processed_image)

    # Run inference
    interpreter.invoke()

    # Get output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Process the output
    predicted_class = np.argmax(output_data[0])
    confidence = float(output_data[0][predicted_class])

    return jsonify({
        'predicted_class': class_names[predicted_class],
        'confidence': confidence
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')