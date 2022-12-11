# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 19:04:30 2022

@author: jstur2828
"""

import tensorflow as tf
from tensorflow import keras

class StopOnAccuracy(tf.keras.callbacks.Callback):
    def __init__(self, accuracy):
        self.accuracy = accuracy

    def on_epoch_end(self, epoch, logs):
        if logs["val_accuracy"] >= self.accuracy:
            self.model.stop_training = True

# Create a StopOnAccuracy callback that stops training when the testing accuracy reaches 99%
#stop_on_accuracy = StopOnAccuracy(0.99)
