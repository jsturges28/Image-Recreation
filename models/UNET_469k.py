# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 19:55:00 2022

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

#combined.reshape((-1, height, width, 1))/255

rand_index = np.random.randint(0,len(combined))
c,s = combined[rand_index], segmented[rand_index]
plt.figure(figsize=(5,5))
plt.imshow(c)
plt.show()

for i in range(10):
    plt.figure(figsize=(5,5))
    plt.imshow(s[:,:,i])
    plt.title(i)
    plt.show()

def build(height, width, c_size):
    
    input_layer = keras.layers.Input(shape=[height, width, c_size])
    
    #padding1 = ZeroPadding2D(padding=11, input_shape=[height, width, c_size])(input_layer)
    
    conv1 = Convolution2D(32, (3,3), padding="same", activation='relu')(input_layer)
    conv2 = Convolution2D(32, (3,3), padding="same", activation='relu')(conv1)
    maxpool1 = MaxPooling2D(pool_size=(2,2))(conv2)
    
    conv3 = Convolution2D(64, (3,3), padding="same", activation='relu')(maxpool1)
    conv4 = Convolution2D(64, (3,3), padding="same", activation='relu')(conv3)
    maxpool2 = MaxPooling2D(pool_size=(2,2))(conv4)
    
    conv5 = Convolution2D(128, (3,3), padding="same", activation='relu')(maxpool2)
    conv6 = Convolution2D(128, (3,3), padding="same", activation='relu')(conv5)                            
    up_conv1 = Conv2DTranspose(filters=64, strides=(2,2), kernel_size=(2,2))(conv6)
    
    for i in range(1,3):
        if (K.int_shape(conv4)[1] == K.int_shape(up_conv1)[1] and K.int_shape(conv4)[2] == K.int_shape(up_conv1)[2]):
            merge1 = tf.keras.layers.concatenate([conv4, up_conv1], axis=-1)
            break
        else:
            diff = K.int_shape(conv4)[i] - K.int_shape(up_conv1)[i]
            while(diff):
                if diff < 0 and i == 1:
                    pad_width = abs(int(ceil(diff/2)))
                    if pad_width == 0:
                        pad_width = 1
                    conv4 = ZeroPadding2D(padding=((0,pad_width),(0,0)))(conv4)
                    diff = K.int_shape(conv4)[i] - K.int_shape(up_conv1)[i]
                
                if diff < 0 and i == 2:
                    pad_width = abs(int(ceil(diff/2)))
                    if pad_width == 0:
                        pad_width = 1
                    conv4 = ZeroPadding2D(padding=((0,0),(0,pad_width)))(conv4)
                    diff = K.int_shape(conv4)[i] - K.int_shape(up_conv1)[i]
            
                if diff > 0 and i == 1:
                    conv4 = Cropping2D(cropping=((1,0), (0,0)))(conv4)
                    diff = K.int_shape(conv4)[i] - K.int_shape(up_conv1)[i]
                elif diff > 0 and i == 2:
                    conv4 = Cropping2D(cropping=((0,0), (0,1)))(conv4)
                    diff = K.int_shape(conv4)[i] - K.int_shape(up_conv1)[i]
                    
    merge1 = tf.keras.layers.concatenate([conv4, up_conv1], axis=-1)
    
    conv7 = Convolution2D(64, (3,3), padding="same", activation='relu')(merge1)
    conv8 = Convolution2D(64, (3,3), padding="same", activation='relu')(conv7)                             
    up_conv2 = Conv2DTranspose(filters=32, strides=(2,2), kernel_size=(2,2))(conv8)
    
    for i in range(1,3):
        if (K.int_shape(conv2)[1] == K.int_shape(up_conv2)[1] and K.int_shape(conv2)[2] == K.int_shape(up_conv2)[2]):
            merge2 = tf.keras.layers.concatenate([conv2, up_conv2], axis=-1)
            break
        else:
            diff = K.int_shape(conv2)[i] - K.int_shape(up_conv2)[i]
            while(diff):
                if diff < 0 and i == 1:
                    pad_width = abs(int(ceil(diff/2)))
                    if pad_width == 0:
                        pad_width = 1
                    conv2 = ZeroPadding2D(padding=((0,pad_width),(0,0)))(conv2)
                    diff = K.int_shape(conv2)[i] - K.int_shape(up_conv2)[i]
                    
                if diff < 0 and i == 2:
                    pad_width = abs(int(ceil(diff/2)))
                    if pad_width == 0:
                        pad_width = 1
                    conv2 = ZeroPadding2D(padding=((0,0),(0,pad_width)))(conv2)
                    diff = K.int_shape(conv2)[i] - K.int_shape(up_conv2)[i]
                
            
                if diff > 0 and i == 1:
                    conv2 = Cropping2D(cropping=((1,0), (0,0)))(conv2)
                    diff = K.int_shape(conv2)[i] - K.int_shape(up_conv2)[i]
                elif diff > 0 and i == 2:
                    conv2 = Cropping2D(cropping=((0,0), (0,1)))(conv2)
                    diff = K.int_shape(conv2)[i] - K.int_shape(up_conv2)[i]
                    
    merge2 = tf.keras.layers.concatenate([conv2, up_conv2], axis=-1)
    
    conv9 = Convolution2D(32, (3,3), padding="same", activation='relu')(merge2)
    conv10 = Convolution2D(32, (3,3), padding="same", activation='relu')(conv9)                          
    
    conv11 = Convolution2D(11, 1, padding="same")(conv10)
    
    return keras.Model(inputs=input_layer,outputs=conv11)

model = build(height, width, 1)
#print(model.summary())
model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(0.001), metrics=['accuracy'])
model.fit(combined, segmented, batch_size=25, epochs=5)