# -*- coding: utf-8 -*-
"""Created on Wed Mar 15 10:09:24 2017"""
import math
import os
import numpy as np
from matplotlib import pyplot as plt
from six.moves import cPickle
from skimage import io
from skimage import transform as tf
from skimage.transform import resize

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
    X=[np.reshape(x, (1024,)) for x in X] #reshape list (1024,) is crucial.
    print(np.shape(X))
    print('data',X)
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
#Load CIFAR10 images
class CifarLoader():
    """IMPORTANT: The data array contains 3072 bytes, where the first 1024 are for
    red, the next green and the last blue.
    """
    def __init__(self, path, **kwargs):
        f = open(path, 'rb')
        datadict = cPickle.load(f,encoding='latin1')
        f.close()
        X = datadict["data"]
        #print(X)
        Y = datadict['labels']
        X = X.reshape(10000,3, 32, 32).astype("float")
        self.X = X
        Y_Neuron = Y
        Y = np.array(Y)

        #------------------------------------------------------------------
        #Separate Cats and Dogs, get indices of the animals
        cat_label = 3 #cifar labels cat with 3
        cat_indices = []

        for i in range(0,len(Y)):
            if Y[i] == cat_label:
                cat_indices.append(i)

        dog_label = 5 #cifar labels dog with 5
        dog_indices = []

        for i in range(0,len(Y)):
            if Y[i] == dog_label:
                dog_indices.append(i)

        #make lists of cat and dog images
        #and list of the labels
        #to be able to generalize to other cifar images I relabel 3 --> [0,0,0,1,0,0...]
        #and 5 to [0,0,0,0,0,1,0..]
        self.cats = []
        self.dogs = []
        self.cats_label = []
        self.dogs_label = []
        for i in cat_indices:
            self.cats.append(X[i])
            self.cats_label.append([0,0,0,1,0,0,0,0,0,0])
        for i in dog_indices:
            self.dogs.append(X[i])
            self.dogs_label.append([0,0,0,0,0,1,0,0,0,0])
