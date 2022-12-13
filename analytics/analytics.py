# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 20:46:23 2022

@author: jstur2828
"""
import math
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import cv2
import time
import csv
import argparse
import re
import sys

# Append paths 

sys.path.append(os.getcwd()) # if we're in the main directory
sys.path.append(os.path.dirname(os.getcwd())) # if we're in the models folder

# Go one level up if we're in the models folder 

if 'analytics' in os.getcwd():
    os.chdir('../')

# Assign path to cwd
path = os.getcwd()

fig_path = os.path.join(path, "figures")

def get_best_value(path, folder):
    files = [f for f in os.listdir(os.path.join(path, folder))]
    vanillaCNNs = []
    UNETs = []
    UNET3p = []
    
    for file in files:
        file_matches1 = re.findall(r'.*VanillaCNN.*', file)
        vanillaCNNs.extend(file_matches1)
    vanillaCNNs.sort()
    
    for file in files:
        file_matches2 = re.findall(r'.*UNET.*', file)
        UNETs.extend(file_matches2)
    UNETs.sort()
    
    for file in files:
        file_matches3 = re.findall(r'.*UNET3p.*', file)
        UNET3p.extend(file_matches3)
    UNET3p.sort()
    
    idx_best_v = 0
    iou_best_v = 0.0
    f_best = ''
    for f in vanillaCNNs:
        with open("%s/%s"%(os.path.join(path, folder),f), "rb") as fp:
            history = pickle.load(fp)
            if history['val_mean_io_u'][-1] > iou_best_v:
                iou_best_v = history['val_mean_io_u'][-1]
                idx_best_v = int(re.search(r'\d+', f).group())
                f_best = f
                
    idx_best_u = 0
    iou_best_u = 0.0
    u_best = ''
    for f in UNETs:
        with open("%s/%s"%(os.path.join(path, folder),f), "rb") as fp:
            history = pickle.load(fp)
            if history['val_mean_io_u'][-1] > iou_best_u:
                iou_best_u = history['val_mean_io_u'][-1]
                idx_best_u = int(re.search(r'\d+', f).group())
                u_best = f
                
    idx_best_u3p = 0
    iou_best_u3p = 0.0
    u3p_best = ''
    for f in UNET3p:
        with open("%s/%s"%(os.path.join(path, folder),f), "rb") as fp:
            history = pickle.load(fp)
            if history['val_mean_io_u'][-1] > iou_best_u3p:
                iou_best_u3p = history['val_mean_io_u'][-1]
                idx_best_u3p = int(re.search(r'\d+', f).group())
                u3p_best = f
                
    #return iou_best_v, idx_best_v, iou_best_u, idx_best_u, iou_best_u3p, idx_best_u3p
    return f_best, u_best, u3p_best

def get_max_value(path, folder):
    files = [f for f in os.listdir(os.path.join(path, folder))]
    vanillaCNNs = []
    UNETs = []
    
    for file in files:
        file_matches1 = re.findall(r'.*VanillaCNN.*', file)
        vanillaCNNs.extend(file_matches1)
    vanillaCNNs.sort()
    
    for file in files:
        file_matches2 = re.findall(r'.*UNET.*', file)
        UNETs.extend(file_matches2)
    UNETs.sort()
    
    
    vals1 = []
    for f in vanillaCNNs:
        with open("%s/%s"%(os.path.join(path, folder),f), "rb") as fp:
            history = pickle.load(fp)
            vals1.append(history['val_mean_io_u'][-1])
                
    vals2 = []
    for f in UNETs:
        with open("%s/%s"%(os.path.join(path, folder),f), "rb") as fp:
            history = pickle.load(fp)
            vals2.append(history['val_mean_io_u'][-1])
                
    #return iou_best_v, idx_best_v, iou_best_u, idx_best_u, iou_best_u3p, idx_best_u3p
    return max(vals1), np.average(vals1), max(vals2), np.average(vals2)
            
def display_iou_bests(path, folder):
    '''
    Plot the learning curves for a set of results
    
    :param base: Directory containing a set of results files
    '''
    matches = []
    bests = get_best_value(path, folder)
    for i in range(len(bests)):
        if len(bests[i]) != 0:
            matches.append(bests[i])
   
    #files = [f for f in os.listdir(os.path.join(path, folder))]
        
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
    plt.legend(matches, fontsize='small')
    
print(get_max_value(path, 'results'))
display_iou_bests(path, 'results')
plt.savefig(os.path.join(fig_path,'results_bests.png'))
plt.close()
