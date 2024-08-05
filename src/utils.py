import numpy as np
import cv2
import base64

def get_image_from_request(request):
    if 'file' in request.files:
        file = request.files['file']
        image_array = np.frombuffer(file.read(), np.uint8)
        return cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    elif 'image' in request.json:
        base64_string = request.json['image']['base64']
        image_data = base64.b64decode(base64_string)
        image_array = np.frombuffer(image_data, np.uint8)
        return cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return None