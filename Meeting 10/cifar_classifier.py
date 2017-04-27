'''Very Simple Cifar Classifier
'''

# System
from __future__ import print_function
import numpy as np
import os
from PIL import Image
from Load_CIFAR import loader

#Keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.datasets import cifar10
#from keras.regularizers import WeightRegularizer, ActivityRegularizer 
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization 
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import BaseLogger, Callback, CSVLogger 

from matplotlib import pyplot as plt

import math
import os
from matplotlib import pyplot as plt
import numpy as np
from six.moves import cPickle 

class classifier():
    def __init__(self, **kwargs):

        cats = []
        dogs = []
        cat_lab = []
        dog_lab = []
        #load all batches
        for i in range(1,nr_batch+1):
            new_data = loader('CIFAR10/Images/data_batch_'+str(i))
            cats = new_data.cats + cats
            dogs = new_data.dogs + dogs
            cat_lab = new_data.cats_label + cat_lab
            dog_lab = new_data.dogs_label + dog_lab
            
        X_train = np.array(cats + dogs)
        Y_train = np.array(cat_lab + dog_lab)

        #----------------------------------------------------------------
        #Neural Network

        from keras import backend as K
        K.set_image_dim_ordering('th')
        model = Sequential()

        model.add(Convolution2D(32, 3, 3, border_mode='same',input_shape=(img_channels, img_rows, img_cols)))
        model.add(Activation('relu'))
        model.add(Convolution2D(32, 3, 3))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        #model.add(Dropout(0.25))

        model.add(Convolution2D(64, 3, 3, border_mode='same'))
        model.add(Activation('relu'))
        model.add(Convolution2D(64, 3, 3))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(nb_classes))
        model.add(Activation('softmax'))

        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss=loss_function,
                      optimizer=sgd,
                      metrics=['accuracy'])

        X_train = X_train.astype('float32')

        X_train /= 255
        model.fit(X_train, Y_train,
                    batch_size=32, epochs=nb_epoch)
        model.save('my_model_relabel_.h5')
        model.save_weights('my_model_weights_relabel_.h5')
#-----------------------------------------------------------------------------
        
loss_function = 'categorical_crossentropy'

nr_batch = 1 #how many batches of cifar shall be included (1 to 5 possible)
nb_classes = 10
nb_epoch = 100
data_augmentation = True

# input image dimensions
img_rows, img_cols = 32, 32
# the CIFAR10 images are RGB
img_channels = 3
        
classifier()





