# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 20:13:10 2022

@author: jstur2828
"""

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
import tensorflow.keras.activations as activations
import tensorflow.keras.metrics as metrics
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, MaxPooling2D, Flatten, \
    TimeDistributed, Convolution2D, Activation, Dropout, LSTM, \
    Convolution3D, MaxPooling3D, Conv2DTranspose, Conv3DTranspose, \
    Attention, ZeroPadding2D, ConvLSTM2D, BatchNormalization, Concatenate, \
    Cropping2D
import numpy as np
import keras.backend as K
from math import ceil, floor 

train, test = (.9, .1)
segmented = np.load('.\\segmented\\segmented.npy')
_, height, width, c_size = segmented.shape

combined=np.load('.\\combined\\combined.npy')

train_data = combined[:int(len(combined)*train),:,:]
train_label = segmented[:int(len(combined)*train),:,:,:]
test_data = combined[int(len(combined)*train):,:,:]
test_label = segmented[int(len(combined)*train):,:,:,:]


def encoder_block(inputs, n_filters, kernel_size, strides):
    encoder = ZeroPadding2D(padding=5, input_shape=inputs.shape)(inputs)
    encoder = Convolution2D(filters=n_filters, kernel_size=kernel_size, strides=strides, padding='valid', use_bias=False)(encoder)
    encoder = BatchNormalization()(encoder)
    encoder = Activation(activations.gelu)(encoder)
    encoder = Convolution2D(filters=n_filters, kernel_size=kernel_size, padding='valid', use_bias=False)(encoder)
    encoder = BatchNormalization()(encoder)
    encoder = Activation(activations.gelu)(encoder)
    #encoder = MaxPooling2D(pool_size=(2,2), strides=(2,2)))(encoder)
    return encoder

def upscale_blocks(inputs):
    n_upscales = len(inputs)
    upscale_layers = []
    
    for i, inp in enumerate(inputs):
        p = n_upscales - i
        u = Conv2DTranspose(filters=64, kernel_size=3, strides=2**p, padding='valid')(inp)
        
        for i in range(2):
            u = Convolution2D(filters=64, kernel_size=3, padding='valid', use_bias=False)(u)
            u = BatchNormalization()(u)
            u = Activation(activations.gelu)(u)
            
        upscale_layers.append(u)
    return upscale_layers

def decoder_block(layers_to_upscale, inputs):
    upscaled_layers = upscale_blocks(layers_to_upscale)

    decoder_blocks = []

    for i, inp in enumerate(inputs):
        d = Convolution2D(filters=64, kernel_size=3, strides=2**i, padding='valid', use_bias=False)(inp)
        d = BatchNormalization()(d)
        d = Activation(activations.gelu)(d)
        d = Convolution2D(filters=64, kernel_size=3, padding='valid', use_bias=False)(d)
        d = BatchNormalization()(d)
        d = Activation(activations.gelu)(d)

        decoder_blocks.append(d)
        
    '''
    This try/except block is to make sure the concatenations operate smoothly.
    All blocks to be merged are compared with the final decoder block in the list, and fit to those dimensions.
    '''
    
    try:
        decoder = layers.concatenate(upscaled_layers + decoder_blocks)
    except ValueError:
        #print("A `Concatenate` layer requires inputs with matching shapes except for the concatenation axis.")
        pass
    finally:
        for i in range(0, len(upscaled_layers)):
            for j in range(1,3):
            
                diff = K.int_shape(upscaled_layers[i])[j] - K.int_shape(decoder_blocks[-1])[j]
                while(diff):
                    if diff < 0:
                        pad_width = abs(int(ceil(diff/2)))
                        if pad_width == 0:
                            pad_width = 1
                        if j == 1:
                            upscaled_layers[i] = ZeroPadding2D(padding=((0,pad_width),(0,0)))(upscaled_layers[i])
                            diff = K.int_shape(upscaled_layers[i])[j] - K.int_shape(decoder_blocks[-1])[j]
                        if j == 2:
                            upscaled_layers[i] = ZeroPadding2D(padding=((0,0),(0,pad_width)))(upscaled_layers[i])
                            diff = K.int_shape(upscaled_layers[i])[j] - K.int_shape(decoder_blocks[-1])[j]
                
                    if diff > 0 and j == 1:
                        upscaled_layers[i] = Cropping2D(cropping=((1,0), (0,0)))(upscaled_layers[i])
                        diff = K.int_shape(upscaled_layers[i])[j] - K.int_shape(decoder_blocks[-1])[j]
                        
                    elif diff > 0 and j == 2:
                        upscaled_layers[i] = Cropping2D(cropping=((0,0), (0,1)))(upscaled_layers[i])
                        diff = K.int_shape(upscaled_layers[i])[j] - K.int_shape(decoder_blocks[-1])[j]
                
        for i in range(0, len(decoder_blocks)):
            for j in range(1,3):
                diff = K.int_shape(decoder_blocks[i])[j] - K.int_shape(decoder_blocks[-1])[j]
                while(diff):
                    if diff < 0:
                        pad_width = abs(int(ceil(diff/2)))
                        if pad_width == 0:
                            pad_width = 1
                        if j == 1:
                            decoder_blocks[i] = ZeroPadding2D(padding=((0,pad_width),(0,0)))(decoder_blocks[i])
                            diff = K.int_shape(decoder_blocks[i])[j] - K.int_shape(decoder_blocks[-1])[j]
                        if j == 2:
                            decoder_blocks[i] = ZeroPadding2D(padding=((0,0),(0,pad_width)))(decoder_blocks[i])
                            diff = K.int_shape(decoder_blocks[i])[j] - K.int_shape(decoder_blocks[-1])[j]
                
                    if diff > 0 and j == 1:
                        decoder_blocks[i] = Cropping2D(cropping=((1,0), (0,0)))(decoder_blocks[i])
                        diff = K.int_shape(decoder_blocks[i])[j] - K.int_shape(decoder_blocks[-1])[j]
                        
                    elif diff > 0 and j == 2:
                        decoder_blocks[i] = Cropping2D(cropping=((0,0), (0,1)))(decoder_blocks[i])
                        diff = K.int_shape(decoder_blocks[i])[j] - K.int_shape(decoder_blocks[-1])[j]

                
                
    decoder = layers.concatenate(upscaled_layers + decoder_blocks)
    decoder = Convolution2D(filters=256, kernel_size=3, padding='valid', use_bias=False)(decoder)
    decoder = BatchNormalization()(decoder)
    decoder = Activation(activations.gelu)(decoder)

    return decoder

def build(height, width, c_size):
    inputs = keras.layers.Input(shape=[height, width, c_size])
    
    e1 = encoder_block(inputs, n_filters=32, kernel_size=3, strides=1)
    e2 = encoder_block(e1, n_filters=64, kernel_size=3, strides=2)
    e3 = encoder_block(e2, n_filters=128, kernel_size=3, strides=2)
    #e4 = encoder_block(e3, n_filters=256, kernel_size=3, strides=2)
    #e5 = encoder_block(e4, n_filters=512, kernel_size=3, strides=2)

    #d4 = decoder_block(layers_to_upscale=[e5], inputs=[e4, e3, e2, e1])
    #d3 = decoder_block(layers_to_upscale=[e5, d4], inputs=[e3, e2, e1])
    #d2 = decoder_block(layers_to_upscale=[e5, d4, d3], inputs=[e2, e1])
    #d1 = decoder_block(layers_to_upscale=[e5, d4, d3, d2], inputs=[e1])
    
    #d3 = decoder_block(layers_to_upscale=[e4], inputs=[e3, e2, e1])
    #d2 = decoder_block(layers_to_upscale=[e4, d3], inputs=[e2, e1])
    #d1 = decoder_block(layers_to_upscale=[e4, d3, d2], inputs=[e1])
    
    d2 = decoder_block(layers_to_upscale=[e3], inputs=[e2, e1])
    d1 = decoder_block(layers_to_upscale=[e3, d2], inputs=[e1])
    
    
    
    output = Convolution2D(filters=11, kernel_size=1, padding='valid', activation='tanh')(d1)
    output_final = Activation("softmax")(output)
    
    model = models.Model(inputs, output_final)
    
    return model

#model = build(10, 32, 32, 1)
model = build(height, width, 1)
#model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(0.001), metrics=['accuracy'])
#model.fit(combined, segmented, batch_size=25, epochs=5)

print(model.summary())

        
        
