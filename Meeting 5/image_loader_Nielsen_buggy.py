# -*- coding: utf-8 -*-
"""Created on Wed Mar 15 10:09:24 2017"""
import math
import os
from skimage import io
from skimage import transform as tf
from matplotlib import pyplot as plt
from skimage.transform import resize
import numpy as np

#----------------------------------------------------------------------------
#Load Images
def load_images(folder):
    images = []
    labels = []
    for file in os.listdir(folder):
        if file.endswith(".png"):
            images.append(io.imread(folder + file))
        if file.find("einstein") > -1 or file.find("curie") > -1:
            labels.append(1)
        else:
            labels.append(0)
    return images, labels

#----------------------------------------------------------------------------
#Resize Images    
def prep_datas(xset,xlabels):
    X=list(xset)
    for i in range(len(X)):
        X[i]=resize(X[i],(32,32,1)) #reduce the size of the image from 100X100 to 32X32. Also flattens the color levels
    print(np.shape(X))
    X=[np.reshape(x, (1024, 1)) for x in X] # reshape the liste to have the form required by keras (theano), ie (1,32,32)
    print(np.shape(X))
    Y = xlabels
    return X,Y

#----------------------------------------------------------------------------
#Deshear function to make pictures "straight"
def deshear(filename):
    image = io.imread(filename)
    distortion = image.shape[1] - image.shape[0]
    shear = tf.AffineTransform(shear=math.atan(distortion/image.shape[0]))
    return tf.warp(image, shear)[:, distortion:]

#-----------------------------------------------------------------------------

