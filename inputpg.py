import cv2
import numpy as np
import tensorflow as tf
# Load the trained model
model = tf.keras.models.load_model(r"your trained model")
class_labels = {0: 'non_disease', 1: 'disease'}
input_image_path = r"your test image"
input_image = cv2.imread(input_image_path)
input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
input_image = cv2.resize(input_image, (224, 224))
input_image = tf.keras.applications.resnet50.preprocess_input(input_image)
input_image = np.expand_dims(input_image, axis=0)
predictions = model.predict(input_image)
predicted_class = np.argmax(predictions)
confidence = np.max(predictions)

predicted_label = class_labels[predicted_class]
result_text = f"Class: {predicted_label}, Confidence: {confidence:.2f}"
print(result_text)
output_image = cv2.cvtColor(input_image[0], cv2.COLOR_RGB2BGR)
output_image = cv2.putText(output_image, result_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)    
cv2.imshow('Classification Result', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
