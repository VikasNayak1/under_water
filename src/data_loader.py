# data_loader.py

import cv2
import numpy as np
import os

IMAGE_HEIGHT, IMAGE_WIDTH = 256, 256

def data_generator(input_image_paths, output_image_paths, batch_size):
    while True:
        for i in range(0, len(input_image_paths), batch_size):
            batch_input_paths = input_image_paths[i:i+batch_size]
            batch_output_paths = output_image_paths[i:i+batch_size]

            input_images = [cv2.resize(cv2.imread(img_path), (IMAGE_WIDTH, IMAGE_HEIGHT)) / 255.0 for img_path in batch_input_paths]
            output_images = [cv2.resize(cv2.imread(img_path), (IMAGE_WIDTH, IMAGE_HEIGHT)) / 255.0 for img_path in batch_output_paths]
            
            yield np.array(input_images), np.array(output_images)
