import os
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from model import underwater_enhancement_cnn
import glob


input_image_paths = sorted(glob.glob('data/input_images/*.jpg'))
output_image_paths = sorted(glob.glob('data/output_images/*.jpg'))


BATCH_SIZE = 4
EPOCHS = 50

def load_image(image_path):
    
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [256, 256])
    image = tf.cast(image, tf.float32) / 255.0 
    return image

def create_dataset(input_image_paths, output_image_paths, batch_size):
    
    input_images = tf.data.Dataset.from_tensor_slices(input_image_paths)
    output_images = tf.data.Dataset.from_tensor_slices(output_image_paths)
    
    
    input_images = input_images.map(lambda x: load_image(x))
    output_images = output_images.map(lambda x: load_image(x))
    
   
    dataset = tf.data.Dataset.zip((input_images, output_images))
    
   
    dataset = dataset.shuffle(buffer_size=len(input_image_paths))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)  
    
    return dataset


model = underwater_enhancement_cnn()
model.compile(optimizer=Adam(), loss='mse', metrics=['mae'])

train_dataset = create_dataset(input_image_paths, output_image_paths, BATCH_SIZE)

model.fit(train_dataset, epochs=EPOCHS, steps_per_epoch=len(input_image_paths) // BATCH_SIZE)

model.save("saved_models/underwater_enhancement_cnn.h5")
