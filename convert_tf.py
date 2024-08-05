import tensorflow as tf

# Load your .h5 model
model = tf.keras.models.load_model('model/mobilenet_v2.h5')

# Convert the model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model
with open('model/mobilenet_v2.tflite', 'wb') as f:
    f.write(tflite_model)