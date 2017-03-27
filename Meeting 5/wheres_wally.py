# -*- coding: utf-8 -*-
"""Created on Wed Mar 15 10:09:24 2017"""
import math
import os
import numpy as np
from skimage import io
from skimage import transform as tf
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.optimizers import SGD
from keras.utils import np_utils
from matplotlib import pyplot as plt


def load_images(folder):
    images = []
    labels = []
    for file in os.listdir(folder):
        if file.endswith(".png"):
            images.append(io.imread(folder + file, as_grey=True))
        if file.find("einstein") > -1:
            labels.append(1)
        elif file.find("curie") > -1:
            labels.append(2)
        else:
            labels.append(0)
    return images, labels


def deshear(filename):
    image = io.imread(filename)
    distortion = image.shape[1] - image.shape[0]
    shear = tf.AffineTransform(shear=math.atan(distortion/image.shape[0]))
    return tf.warp(image, shear)[:, distortion:]


def normalize_images(images):
    for i in range(len(images)):
        images[i] = images[i]/np.amax(images[i])
    return images


class LeNet:

    def __init__(self, input_shape, conv_1, pool_1, conv_2, pool_2, hidden,
                 classes):
        self.model = Sequential()
        # first set of CONV => RELU => POOL
        self.model.add(Conv2D(*conv_1, padding='same', activation='relu',
                              data_format='channels_last',
                              input_shape=input_shape))
        self.model.add(MaxPooling2D(pool_1[0], pool_1[1]))
        # second set of CONV => RELU => POOL
        self.model.add(Conv2D(*conv_2, padding='same', activation='relu',
                              data_format='channels_last'))
        self.model.add(MaxPooling2D(pool_2[0], pool_2[1]))
        # set of FC => RELU layers
        self.model.add(Flatten())
        self.model.add(Dense(hidden, activation='relu'))
        # softmax classifier
        self.model.add(Dense(classes, activation='softmax'))

training_set, training_labels = load_images("images/train/")
test_set, test_labels = load_images("images/test/")
training_set = normalize_images(np.array(training_set,float))
training_set = training_set[..., np.newaxis]
test_set = normalize_images(np.array(test_set,float))
test_set = test_set[..., np.newaxis]

classes = 3  # number of classes to identify
hidden = 500  # number of nuerons in hidden layer
conv_1 = (20, (15, 15))  # (num of filters in first layer, filter size)
conv_2 = (50, (15, 15))  # (num of filters in second layer, filter size)
pool_1 = ((6, 6), (6, 6))  # (size of pool matrix, stride)
pool_2 = ((6, 6), (6, 6))  # (size of pool matrix, stride)
training_labels = np_utils.to_categorical(training_labels, classes)
test_labels = np_utils.to_categorical(test_labels, classes)

aps = LeNet(training_set[1].shape, conv_1, pool_1, conv_2, pool_2, hidden,
            classes)

aps.model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=["accuracy"])
aps.model.fit(training_set, training_labels, batch_size=10, epochs=50,
              verbose=1)

(loss, accuracy) = aps.model.evaluate(test_set, test_labels, batch_size=6, verbose=1)
probs = aps.model.predict(test_set)
prediction = probs.argmax(axis=1)

