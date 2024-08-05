import numpy as np
import cv2
import tflite_runtime.interpreter as tflite

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

# class ObjectDetector:
#     def __init__(self, model_path, labels_path):
#         self.interpreter = tflite.Interpreter(model_path=model_path)
#         self.interpreter.allocate_tensors()
#         self.input_details = self.interpreter.get_input_details()
#         self.output_details = self.interpreter.get_output_details()
#         self.class_names = self.read_class_names(labels_path)

#     def read_class_names(self, file_path):
#         with open(file_path, 'r') as f:
#             return [line.strip() for line in f.readlines()]

#     def preprocess_image(self, image):
#         input_shape = self.input_details[0]['shape'][1:3]
#         image = cv2.resize(image, input_shape)
#         image = image.astype(np.float32) / 255.0
#         image = np.expand_dims(image, axis=0)
#         return image

#     def postprocess_output(self, boxes, scores, classes, count):
#         results = []
#         for i in range(int(count[0])):
#             if scores[0][i] > 0.5:  # Confidence threshold
#                 ymin, xmin, ymax, xmax = boxes[0][i]
#                 results.append({
#                     'class': self.class_names[int(classes[0][i])],
#                     'confidence': float(scores[0][i]),
#                     'bounding_box': {
#                         'xmin': float(xmin),
#                         'ymin': float(ymin),
#                         'xmax': float(xmax),
#                         'ymax': float(ymax)
#                     }
#                 })
#         return results

#     def detect(self, image):
#         processed_image = self.preprocess_image(image)
#         self.interpreter.set_tensor(self.input_details[0]['index'], processed_image)
#         self.interpreter.invoke()
#         boxes = self.interpreter.get_tensor(self.output_details[0]['index'])
#         classes = self.interpreter.get_tensor(self.output_details[1]['index'])
#         scores = self.interpreter.get_tensor(self.output_details[2]['index'])
#         count = self.interpreter.get_tensor(self.output_details[3]['index'])
#         return self.postprocess_output(boxes, scores, classes, count)