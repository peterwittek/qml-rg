import math
import os
from matplotlib import pyplot as plt
import numpy as np
from six.moves import cPickle



"""IMPORTANT: The data array contains 3072 bytes, where the first 1024 are for
red, the next green and the last blue.
"""

class Loader():


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
