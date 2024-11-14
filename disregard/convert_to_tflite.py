import tensorflow as tf

# Load the model from the .keras file saved in your Jupyter notebook
model = tf.keras.models.load_model('model/keypoint_classifier/keypoint_classifier.keras')

# Define the path to save the TFLite model
tflite_save_path = 'model/keypoint_classifier/keypoint_classifier.tflite'

# Set up the TFLite converter with the original code
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quantized_model = converter.convert()

# Save the converted model
with open(tflite_save_path, 'wb') as f:
    f.write(tflite_quantized_model)

print()
print(f"TFLite model saved at: {tflite_save_path}")