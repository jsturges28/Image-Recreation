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
from keras.optimizers import Adam
from keras.callbacks import History, EarlyStopping
    
import numpy as np
import keras.backend as K
import math
from math import ceil, floor 
import matplotlib.pyplot as plt
import pickle
import os
import cv2
import time
import csv
import argparse
import re
import sys

# Append paths so python can find our callback class

sys.path.append(os.getcwd()) # if we're in the main directory
sys.path.append(os.path.dirname(os.getcwd())) # if we're in the models folder 

import callbacks
from callbacks import StopOnAccuracy

# Go one level up if we're in the models folder 

if 'models' in os.getcwd():
    os.chdir('../')

# Assign path to cwd
path = os.getcwd()
model_type = 'UNET3Plus'

'''
Helper functions
'''

# Start at experiment 0. Then, every time this is run, we can increment the index to the next experiment without having to do it manually.

def get_last_exp_index(path, folder):
    if os.path.exists(os.path.join(path, folder)):
        files = [f for f in os.listdir(os.path.join(path, folder))]
        if files:
            matches = []
            digits = []

            for file in files:
                file_matches = re.findall(r'.*VanillaCNN.*', file)
                matches.extend(file_matches)
            matches.sort()
            
            if len(matches) == 0:
                return 0
            
            for match in matches:
                digit_match = re.findall(r'\d+', match)
                digits.extend(digit_match)
            digits.sort()
            
            last_index = digits[-1]
            return int(last_index) + 1
        else:
            return 0
    else: 
        return 0
    
# Create global variable to retrieve the last index in the filesystem, feed it into the parser automatically

last_index = get_last_exp_index(path, 'results')
    
def create_parser():
    '''
    Creates an argument parser for the program to run in command line. Can also run in program if no specification is needed.
    '''
    
    parser = argparse.ArgumentParser(description='Vanilla CNN')
    
    parser.add_argument('--exp', type=int, nargs='?', const=last_index, default=last_index, help='Experimental index')
    parser.add_argument('--epochs', type=int, nargs='?', const=20, default=20, help='Number of training epochs')
    parser.add_argument('--val_acc', type=float, nargs='?', const=0.95, default=0.95, help='Max accuracy before model stops training')
    parser.add_argument('--min_delta', type=float, nargs='?', const=0.005, default=0.005, help='Minimum training accuracy improvement for each epoch')
    parser.add_argument('--no_display', action='store_false', help='Dont display set of learning curve(s) after running experiment(s)')
    #parser.add_argument('--run_exp', type=int, nargs='?', const=1, default=1, help='Select number of times to run experiment')
    parser.add_argument('--batch_size', type=int, nargs='?', const=50, default=50, help='Set size of batch')
    parser.add_argument('--no_results', action='store_false', help='Skip predicting values and dont display the handwritten digits')
    parser.add_argument('--no_verbose', action='store_false', help='Skip the display training progress and dont print results to screen')
    
    return parser

def args2string(args):
    '''
    Translate the current set of arguments
    
    :param args: Command line arguments
    '''
    return "exp_%02d_UNET3Plus"%(args.exp)
    

def display_iou_set(path, folder):
    '''
    Plot the learning curves for a set of results
    
    :param base: Directory containing a set of results files
    '''
    
    files = [f for f in os.listdir(os.path.join(path, folder))]
    if files:
        matches = []

        for file in files:
            file_matches = re.findall(r'.*UNET3Plus.*', file)
            matches.extend(file_matches)
        matches.sort()
        
    plt.figure()
    for f in matches:
        with open("%s/%s"%(os.path.join(path, folder),f), "rb") as fp:
            history = pickle.load(fp)
            #time = pickle.loads(fp.readline())
            for key, value in history.items():
                # iter on both keys and values
                if key.startswith('val_mean_io_u'):
                    iou = key
            plt.ylim(0,1)
            plt.plot(history[iou])
    plt.title("Test IOU vs. Epochs")
    plt.ylabel('IOU score')
    plt.xlabel('Epochs')
    plt.legend(matches, fontsize='small', ncol=2)


parser = create_parser()
args = parser.parse_args()

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

# Create an EarlyStopping callback that stops training when the training accuracy doesn't improve by 0.005 over 2 epochs

callback = EarlyStopping(monitor='accuracy', patience=2, min_delta=args.min_delta)

# Create a StopOnAccuracy callback that stops training when the testing accuracy reaches 93%

stop_on_accuracy = StopOnAccuracy(args.val_acc)

#model = build(10, 32, 32, 1)
model = build(height, width, 1)
print(model.summary())
#model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(0.001), metrics=['accuracy'])
#model.fit(combined, segmented, batch_size=25, epochs=5)

model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(0.001), metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=2)])

argstring = args2string(args)
print("EXPERIMENT: %s"%argstring)

start_time = time.time()
history = model.fit(combined, segmented, validation_split=0.33, batch_size=args.batch_size, epochs=args.epochs, verbose=args.no_verbose, callbacks=[callback, stop_on_accuracy])
end_time = time.time()
tot_time = float("%.2f"%(end_time - start_time))
    
if not os.path.exists(os.path.join(path, "results")):
    os.mkdir(os.path.join(path, "results"))

res_path = os.path.join(path, "results")
    
fp = open("results\\results_%s.pkl"%(argstring), "wb")
pickle.dump(history.history, fp)
#pickle.dump(args, fp)
fp.write(b"\n")
pickle.dump(tot_time, fp)
fp.close()

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
    plt.figure(figsize=(8, 11))
    plt.suptitle('Vanilla CNN image recreations on 3 random images')
    for i in range(N_TEST):
        plt.subplot(4, N_TEST, i+1)
        plt.imshow(originals[i].reshape((HEIGHT, WIDTH)))
        plt.gca().title.set_text("Original image")
        plt.subplot(4, N_TEST, i+1+N_TEST)
        plt.imshow(np.argmax(predicted[i], axis=2))
        plt.gca().title.set_text("Predicted image")
        plt.subplot(4, N_TEST, i+1+2*N_TEST)
        plt.imshow(np.argmax(ground_truth[i], axis=2))
        plt.gca().title.set_text("Ground truth")

if not os.path.exists(os.path.join(path, "figures")):
    os.mkdir(os.path.join(path, "figures"))
fig_path = os.path.join(path, "figures")

# This looks weird, but the boolean argument was set to false as default. So, invoking --no_results will return false and the loop won't run.

if args.no_results:
    print_test(3, 64, 84, combined, segmented, model)
    plt.savefig(os.path.join(fig_path,'predicted_digits_%s.png'%argstring))
    plt.close()

history_dict = history.history

# This looks weird, but the boolean argument was set to false as default. So, invoking --no_display will return false and the loop won't run.
if args.no_display:
    display_iou_set(path, 'results')
    plt.savefig(os.path.join(fig_path,'results_%s.png'%argstring))
    plt.close()

#print("Epoch stopped at epoch", callback.stopped_epoch)

# This looks weird, but the boolean argument was set to false as default. So, invoking --no_verbose will return false and the loop won't run.
if args.no_verbose:
    #get number of epochs
    print("Number of epochs run:", len(history.history['accuracy']))
    print("IOU score:", float("%.4f"%history.history['val_mean_io_u'][-1]))
    print("Time to run model: ", tot_time, "seconds")

        
        
