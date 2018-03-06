from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, load_model

import tensorflow as tf
import cv2
import numpy as np
import os

nbatch = 80

classes = ['None','0','1','2','3','4','5', '6']

train_datagen = ImageDataGenerator( rescale=1./255,
                                    rotation_range=15.,
                                    width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    horizontal_flip=True,
                                  )

test_datagen  = ImageDataGenerator( rescale=1./255 )

train_gen = train_datagen.flow_from_directory(
        'images/my_data/train/',
        target_size=(45, 45),
        color_mode='grayscale',
        batch_size=nbatch,
        classes=classes,
        class_mode='categorical'
    )

test_gen = test_datagen.flow_from_directory(
        'images/my_data/test/',
        target_size=(45, 45),
        color_mode='grayscale',
        batch_size=nbatch,
        classes=classes,
        class_mode='categorical'
    )

model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=(45,45,1)))
BatchNormalization(axis = -1)
model.add(Activation('relu'))
model.add(Conv2D(64,(3,3)))
BatchNormalization(axis = -1)
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128,(3, 3)))
BatchNormalization(axis=-1)
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3, 3)))
BatchNormalization(axis=-1)
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

#Fully connected neural network
model.add(Dense(512))
BatchNormalization()
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(8))
model.add(Activation('softmax'))

model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

callbacks_list = [
    EarlyStopping(monitor='val_loss', patience=10),
    ModelCheckpoint(filepath='model_6cat_rps_2.h5', monitor='val_loss', save_best_only=True),
]

history = model.fit_generator(
        train_gen,
        steps_per_epoch=60,
        epochs=30,
        validation_data=test_gen,
        validation_steps=20,
        callbacks=callbacks_list
    )