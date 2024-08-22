import numpy as np
import cv2
import tflite_runtime.interpreter as tflite
import logging
import onnxruntime as ort

# Konfigurasi logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class ImageClassifier:
    def __init__(self, model_path, labels_path):
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.class_names = self.read_class_names(labels_path)

    def read_class_names(self, file_path):
        with open(file_path, 'r') as f:
            return [line.strip() for line in f.readlines()]

    def preprocess_image(self, image):
        input_shape = self.input_details[0]['shape'][1:3]
        image = cv2.resize(image, input_shape)
        image = image.astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=0)
        return image

    def predict(self, image):
        processed_image = self.preprocess_image(image)
        self.interpreter.set_tensor(self.input_details[0]['index'], processed_image)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        predicted_class = np.argmax(output_data[0])
        confidence = float(output_data[0][predicted_class])
        return self.class_names[predicted_class], confidence

class ObjectDetector:
    def __init__(self, model_path, labels_path):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.class_names = self.read_class_names(labels_path)
        

    def read_class_names(self, file_path):
        with open(file_path, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
        num_classes = self.session.get_outputs()[0].shape[-1] - 5  # Subtract 5 for box coordinates and objectness score
        if len(class_names) < num_classes:
            class_names.extend([f"Unknown_{i}" for i in range(len(class_names), num_classes)])
        return class_names

    def preprocess_image(self, image):
        input_shape = self.session.get_inputs()[0].shape[2:]  # Assuming NCHW format
        image = cv2.resize(image, input_shape)
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))  # HWC to CHW
        image = np.expand_dims(image, axis=0)
        return image

    def detect(self, image):
        processed_image = self.preprocess_image(image)
        output = self.session.run([self.output_name], {self.input_name: processed_image})[0]
        
        # Log raw output
        logger.info("Raw model output shape: %s", output.shape)
        logger.info("Raw model output (first 5 rows):\n%s", output[0, :5, :])
        
        # Calculate and log some statistics
        logger.info("Output statistics:")
        logger.info("Min value: %f", np.min(output))
        logger.info("Max value: %f", np.max(output))
        logger.info("Mean value: %f", np.mean(output))
        logger.info("Median value: %f", np.median(output))
        
        # Process the output tensor
        return self.postprocess_output(output)

    def postprocess_output(self, output):
        results = []
        
        # Reshape output to [8400, 10]
        output = np.squeeze(output).T
        
        # Log reshaped output
        logger.info("Reshaped output (first 5 rows):\n%s", output[:5, :])
        
        # Extract boxes, scores, and class ids
        boxes = output[:, :4]  # First 4 columns are box coordinates
        scores = output[:, 4]  # 5th column is objectness score
        class_ids = output[:, 5:]  # Remaining columns are class probabilities
        
        # Log extracted components
        logger.info("Boxes (first 5):\n%s", boxes[:5, :])
        logger.info("Scores (first 5): %s", scores[:5])
        logger.info("Class IDs (first 5):\n%s", class_ids[:5, :])
        
        # Get indices of detections with score above threshold
        score_threshold = 0.01  # Set a very low threshold for logging
        high_score_indices = np.where(scores > score_threshold)[0]
        
        logger.info("Number of detections above threshold: %d", len(high_score_indices))
        
        for index in high_score_indices:
            class_id = np.argmax(class_ids[index])
            class_score = class_ids[index][class_id]
            confidence = float(scores[index] * class_score)
            
            box = boxes[index]
            
            result = {
                'class': self.class_names[class_id] if class_id < len(self.class_names) else f"Unknown_{class_id}",
                'confidence': confidence,
                'bounding_box': {
                    'xmin': float(box[0]),
                    'ymin': float(box[1]),
                    'xmax': float(box[2]),
                    'ymax': float(box[3])
                }
            }
            results.append(result)
            logger.info("Detection: %s", result)
        
        return results
    
def draw_detections(image, detections):
    for detection in detections:
        box = detection['bounding_box']
        label = f"{detection['class']}: {detection['confidence']:.2f}"

        logger.info("Drawing bounding box : %s", box)
        logger.info("Drawing bounding box and label: %s", label)
        
        # Konversi koordinat relatif menjadi koordinat absolut
        h, w = image.shape[:2]
        x1, y1 = int(box['xmin'] * w), int(box['ymin'] * h)
        x2, y2 = int(box['xmax'] * w), int(box['ymax'] * h)
        
        # Gambar bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Tambahkan label
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return image