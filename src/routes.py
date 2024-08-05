from flask import jsonify, request
from services import ImageClassifier, ObjectDetector
from utils import get_image_from_request
from config import CLASSIFICATION_MODEL_PATH, CLASSIFICATION_LABELS_PATH, DETECTION_MODEL_PATH, DETECTION_LABELS_PATH
import logging

classifier = ImageClassifier(CLASSIFICATION_MODEL_PATH, CLASSIFICATION_LABELS_PATH)
# detector = ObjectDetector(DETECTION_MODEL_PATH, DETECTION_LABELS_PATH)

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

    # @app.route('/detect', methods=['POST'])
    # def detect():
    #     try:
    #         image = get_image_from_request(request)
    #         if image is None:
    #             return jsonify({'error': 'No image provided'}), 400
            
    #         results = detector.detect(image)
    #         app.logger.info(f"Detection results: {len(results)} objects detected")
    #         return jsonify({
    #             'detections': results
    #         })
    #     except Exception as e:
    #         app.logger.error(f"Error in detection: {str(e)}")
    #         return jsonify({'error': 'An error occurred during detection'}), 500