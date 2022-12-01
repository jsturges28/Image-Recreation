# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 15:26:42 2022

@author: jstur2828
"""

import math
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras as keras
import keras.backend as K
import pickle
import os
import cv2

from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, MaxPooling2D, Flatten, \
    Convolution2D, Activation, Dropout, LSTM, \
    Convolution3D, MaxPooling3D, Conv2DTranspose, Conv3DTranspose, \
    Attention, Input, ZeroPadding2D, Cropping2D
from keras.optimizers import Adam
from math import ceil

train, test = (.9, .1)
segmented = np.load('.\\segmented\\segmented.npy')
_, height, width, c_size = segmented.shape

combined=np.load('.\\combined\\combined.npy')

train_data = combined[:int(len(combined)*train),:,:]
train_label = segmented[:int(len(combined)*train),:,:,:]
test_data = combined[int(len(combined)*train):,:,:]
test_label = segmented[int(len(combined)*train):,:,:,:]

def build(height, width, c_size):
    input_layer = keras.layers.Input(shape=[height, width, c_size])
    conv1 = Convolution2D(16, (3,3), padding='same', activation='relu')(input_layer)
    drop1 = Dropout(0.25)(conv1)
    conv2 = Convolution2D(32, (3,3), padding='same', activation='relu')(drop1)
    drop2 = Dropout(0.25)(conv2)
    conv3 = Convolution2D(11, (3,3), padding='same', activation='relu')(drop2)
    act1 = Activation('softmax')(conv3)
    
    return keras.Model(inputs = input_layer, outputs = act1)
    
model = build(height, width, 1)

print(model.summary())

model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(0.001), metrics=['accuracy'])
history = model.fit(combined, segmented, validation_split=0.33, batch_size=50, epochs=30)

def print_test(N_TEST, HEIGHT, WIDTH, combined_test, segmented_test, model):
    rand_index = np.random.randint(0,len(combined_test))
    combined_test=np.reshape(combined_test,(len(combined_test),HEIGHT,WIDTH,1))
    segmented_test=np.reshape(segmented_test,(len(segmented_test),HEIGHT,WIDTH,11))
    originals = combined_test[rand_index:rand_index+N_TEST,:,:,:]
    ground_truth = segmented_test[rand_index:rand_index+N_TEST,:,:,:]
    maxig = np.argmax(ground_truth[0], axis=2)
    predicted = model.predict(originals)
    predicted = np.round(predicted).astype(int)
    maxi = np.argmax(predicted[0], axis=2)
    plt.figure(figsize=(80, 100))
    for i in range(N_TEST):
        plt.subplot(4, N_TEST, i+1)
        plt.imshow(originals[i].reshape((HEIGHT, WIDTH)))
        plt.subplot(4, N_TEST, i+1+N_TEST)
        plt.imshow(np.argmax(predicted[i], axis=2))
        plt.subplot(4, N_TEST, i+1+2*N_TEST)
        plt.imshow(np.argmax(ground_truth[i], axis=2))
        
print_test(3, 64, 84, combined, segmented, model)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

    