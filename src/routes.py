from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
from io import BytesIO
from services import ImageClassifier, ObjectDetector, draw_detections
from utils import get_image_from_request
from config import CLASSIFICATION_MODEL_PATH, CLASSIFICATION_LABELS_PATH, DETECTION_MODEL_PATH, DETECTION_LABELS_PATH
import logging

classifier = ImageClassifier(CLASSIFICATION_MODEL_PATH, CLASSIFICATION_LABELS_PATH)
detector = ObjectDetector(DETECTION_MODEL_PATH, DETECTION_LABELS_PATH)

def configure_routes(app):
    @app.route("/")
    def index():
        return "Image Processing API. Use /predict for classification or /detect for object detection."

    @app.route('/predict', methods=['POST'])
    def predict():
        try:
            image = get_image_from_request(request)
            if image is None:
                return jsonify({'error': 'No image provided'}), 400
            
            predicted_class, confidence = classifier.predict(image)
            app.logger.info(f"Classification result: {predicted_class} with confidence {confidence}")
            return jsonify({
                'predicted_class': predicted_class,
                'confidence': confidence
            })
        except Exception as e:
            app.logger.error(f"Error in prediction: {str(e)}")
            return jsonify({'error': 'An error occurred during prediction'}), 500

    @app.route('/detect', methods=['POST'])
    def detect():
        try:
            image = get_image_from_request(request)
            if image is None:
                return jsonify({'error': 'No image provided'}), 400
            
            results = detector.detect(image)
            app.logger.info(f"Detection results: {len(results)} objects detected")
            return jsonify({
                'detections': results
            })
        except Exception as e:
            app.logger.error(f"Error in detection: {str(e)}", exc_info=True)
            return jsonify({'error': f'An error occurred during detection: {str(e)}'}), 500
    
        
    @app.route('/detect_with_image', methods=['POST'])
    def detect_objects_with_image():
        if 'file' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        image_file = request.files['file']
        image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
        
        detections = detector.detect(image)

        app.logger.info(f"Detection results: {len(detections)} objects detected")
        
        # Gambar bounding box pada gambar
        annotated_image = draw_detections(image, detections)

        app.logger.info(f"Annotated results: {annotated_image}")
        
        # Konversi gambar ke format JPEG
        is_success, buffer = cv2.imencode(".jpg", annotated_image)
        if not is_success:
            return jsonify({'error': 'Failed to encode image'}), 500
        
        # Kirim gambar sebagai response
        return send_file(
            BytesIO(buffer),
            mimetype='image/jpeg',
            as_attachment=True,
            download_name='detected_objects.jpg'
        )