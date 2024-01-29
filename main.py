import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPool2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import opendatasets as od
import keras.utils
import keras.models
import keras.layers

dataset_url = 'https://www.kaggle.com/jutrera/stanford-car-dataset-by-classes-folder'
od.download(dataset_url)

train_dir = './stanford-car-dataset-by-classes-folder/car_data/car_data/train'
validation_dir = './stanford-car-dataset-by-classes-folder/car_data/car_data/test'

img_width, img_height = 224, 224
batch_size = 32
num_classes = 196 

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

# base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))
# base_model.trainable = False

# x = base_model.output
# x = Flatten()(x)
# x = Dense(4096, activation='relu')(x)
# x = Dropout(0.5)(x)
# x = Dense(4096, activation='relu')(x)
# # x = Dense(1024, activation='relu')(x)
# # x = Dense(2048, activation='relu')(x)
# # x = Dense(1024, activation='relu')(x)
# # x = Dense(512, activation='relu')(x)
# # x = Dense(256, activation='relu')(x)
# x = Dropout(0.5)(x)
# predictions = Dense(num_classes, activation='softmax')(x)

# model = Model(inputs=base_model.input, outputs=predictions)

model = tf.keras.models.load_model('vgg16_1.h5') 

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=10, verbose=1, mode='auto')

model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=500,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    callbacks=[checkpoint, early]
)

#model.save('car_type_detection_model_3_layers_12_12_10'+'_v2'+'.h5')
model.save('car_type_detection_model_2_layers_dropout_12_12.h5')