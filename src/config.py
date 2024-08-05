import logging
import os

# Logging configuration
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    filename='app.log',
                    filemode='a')

# Model paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'model')

CLASSIFICATION_MODEL_PATH = os.path.join(MODEL_DIR, "pet_models.tflite")
CLASSIFICATION_LABELS_PATH = os.path.join(MODEL_DIR, "labels.txt")
DETECTION_MODEL_PATH = os.path.join(MODEL_DIR, "object_detection_model.tflite")  # Updated this line
DETECTION_LABELS_PATH = os.path.join(MODEL_DIR, "object_labels.txt")  # Make sure this file exists

# Other configurations
CONFIDENCE_THRESHOLD = 0.5