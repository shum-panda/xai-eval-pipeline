import tensorflow as tf
from tensorflow.keras.applications import resnet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore

print("GPU verfügbar:", tf.config.list_physical_devices('GPU'))

# 1. Modell laden (ImageNet-1K vortrainiert)
model = resnet50.ResNet50(weights='imagenet')
model.trainable = False

# 2. Beispielbild laden
img_path = tf.keras.utils.get_file("elephant.jpg", "https://upload.wikimedia.org/wikipedia/commons/6/63/African_elephant_warning_raised_trunk.jpg")
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
input_tensor = preprocess_input(np.expand_dims(img_array, axis=0))

# 3. Vorhersage anzeigen
preds = model.predict(input_tensor)
decoded = decode_predictions(preds, top=3)[0]
print("Top 3 Predictions:", decoded)

# 4. GradCAM vorbereiten
score = CategoricalScore([np.argmax(preds)])
modifier = ReplaceToLinear()
gradcam = Gradcam(model, model_modifier=modifier)

# 5. Heatmap berechnen
cam = gradcam(score, input_tensor, penultimate_layer='conv5_block3_out')
heatmap = cam[0]

# 6. Visualisierung
def overlay(image, cam, alpha=0.4):
    cam = cv2.resize(cam, (image.shape[1], image.shape[0]))
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlayed = cv2.addWeighted(np.uint8(image), 1 - alpha, heatmap, alpha, 0)
    return overlayed

# Konvertiere das Eingabebild zurück
original_img = np.uint8(img_array)
overlayed_img = overlay(original_img, heatmap)

# Plotten
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(original_img.astype('uint8'))
plt.title("Original")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(overlayed_img)
plt.title("GradCAM")
plt.axis('off')
plt.tight_layout()