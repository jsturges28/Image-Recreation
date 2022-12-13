# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 15:26:42 2022

@author: jstur2828
"""

import math
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
#import matplotlib
import matplotlib.pyplot as plt
import tensorflow.keras as keras
import keras.backend as K
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

import custom_callbacks
from custom_callbacks import StopOnAccuracy

from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, MaxPooling2D, Flatten, \
    Convolution2D, Activation, Dropout, LSTM, \
    Convolution3D, MaxPooling3D, Conv2DTranspose, Conv3DTranspose, \
    Attention, Input, ZeroPadding2D, Cropping2D
from keras.optimizers import Adam
from keras.callbacks import History, EarlyStopping
from math import ceil


# Go one level up if we're in the models folder 

if 'models' in os.getcwd():
    os.chdir('../')

# Assign path to cwd
path = os.getcwd()
model_type = 'VanillaCNN'

'''
Dice and IOU metrics
'''

def dice_coef(y_true, y_pred, smooth=100):        
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice

def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

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
    return "exp_%02d_VanillaCNN"%(args.exp)
    

def display_iou_set(path, folder):
    '''
    Plot the learning curves for a set of results
    
    :param base: Directory containing a set of results files
    '''
    
    files = [f for f in os.listdir(os.path.join(path, folder))]
    if files:
        matches = []

        for file in files:
            file_matches = re.findall(r'.*VanillaCNN.*', file)
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
segmented = np.load('C:\\Users\\jstur2828\\Desktop\\UNET\\segmented\\segmented.npy')
_, height, width, c_size = segmented.shape

combined=np.load('C:\\Users\\jstur2828\\Desktop\\UNET\\combined\\combined.npy')

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

# Create an EarlyStopping callback that stops training when the training accuracy doesn't improve by 0.005 over 2 epochs

callback = EarlyStopping(monitor='accuracy', patience=2, min_delta=args.min_delta)

# Create a StopOnAccuracy callback that stops training when the testing accuracy reaches 93%

stop_on_accuracy = StopOnAccuracy(args.val_acc)
    
model = build(height, width, 1)

print(model.summary())

#tf.keras.metrics.MeanIoU(num_classes=2)

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

'''
train_acc = history_dict["accuracy"]
val_acc = history_dict["val_accuracy"]
train_loss = history_dict["loss"]
val_loss = history_dict["val_loss"]

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
'''

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
    
file = open('vcnn_times.txt')
file.write(str(tot_time)+'\r\n')
file.close()


    