# model.py

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, BatchNormalization, ReLU, Add
from tensorflow.keras.models import Model

def underwater_enhancement_cnn(image_height=256, image_width=256):
    inputs = Input(shape=(image_height, image_width, 3))
    
    x = Conv2D(64, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    outputs = Conv2D(3, (3, 3), padding='same')(x)
    outputs = Add()([inputs, outputs]) 
    
    model = Model(inputs, outputs)
    return model
