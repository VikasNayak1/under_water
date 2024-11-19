import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
import matplotlib.pyplot as plt


model = load_model("saved_models/underwater_enhancement_cnn.h5", custom_objects={'mse': MeanSquaredError()})


def enhance_image(model, img_path):
    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, (256, 256)) / 255.0
    img_resized = np.expand_dims(img_resized, axis=0)
 
    enhanced_img = model.predict(img_resized)[0]
    enhanced_img = np.clip(enhanced_img, 0, 1) 
    
    return (enhanced_img * 255).astype(np.uint8)


test_img_path = 'data/test_images/test13.jpg'


enhanced_image = enhance_image(model, test_img_path)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(cv2.imread(test_img_path), cv2.COLOR_BGR2RGB))

plt.subplot(1, 2, 2)
plt.title("Enhanced Image")
plt.imshow(cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB))
plt.show()
