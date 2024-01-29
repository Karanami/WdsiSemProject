import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPool2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import opendatasets as od
import keras.utils
import keras.models
import keras.layers

working_model = 'car_type_detection_model_4_11_10_9_8_layers.h5'

img_width, img_height = 224, 224
batch_size = 1
num_classes = 196 

predictions_dir = './predictions_dataset/car_data/test'

predictions_datagen = ImageDataGenerator(rescale=1./255)

predictions_generator = predictions_datagen.flow_from_directory(
    predictions_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

model = tf.keras.models.load_model(working_model)

model.summary()

loss, acc = model.evaluate(predictions_generator)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

# print(model.predict(predictions_generator))