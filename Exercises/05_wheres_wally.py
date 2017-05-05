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
from keras.preprocessing.image import ImageDataGenerator
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
            elif os.path.splitext(file)[0].isdigit():
                labels.append(int(os.path.splitext(file)[0]))
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
        images[i] = images[i][0:100, 0:100]
        images[i] = images[i]/np.amax(images[i])
    return np.array(images)


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

# Loading image data sets and normalizing color scale
training_set, training_labels = load_images("images/train/")
test_set, test_labels = load_images("images/test/")
rw_set, rw_file_labels = load_images("images/real_world/")
training_set = normalize_images(training_set)
training_set = training_set[..., np.newaxis]
test_set = normalize_images(test_set)
test_set = test_set[..., np.newaxis]
rw_set = normalize_images(rw_set)
rw_set = rw_set[..., np.newaxis]
rw_set = np.array([x for (y, x) in sorted(zip(rw_file_labels, rw_set))])

# Getting labels for real world set from file
f = open('images/real_world/labels.txt', "r")
lines = f.readlines()
rw_labels = []
for x in lines:
    rw_labels.append(int((x.split('	')[1]).replace('\n', '')))
f.close()

# Parameters for LeNet convolutional network
classes = 3  # number of classes to identify
hidden = 500  # number of nuerons in hidden layer
conv_1 = (20, (15, 15))  # (num of filters in first layer, filter size)
conv_2 = (50, (15, 15))  # (num of filters in second layer, filter size)
pool_1 = ((6, 6), (6, 6))  # (size of pool matrix, stride)
pool_2 = ((6, 6), (6, 6))  # (size of pool matrix, stride)

# Converting integer labels to categorical labels
training_labels = np_utils.to_categorical(training_labels, classes)
test_labels = np_utils.to_categorical(test_labels, classes)
rw_labels = np_utils.to_categorical(rw_labels, classes)

# Creating LeNet from basic training data
aps = LeNet(training_set[1].shape, conv_1, pool_1, conv_2, pool_2, hidden,
            classes)
aps.model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=["accuracy"])
print('\nTraining LeNet with basic training data\n')
aps.model.fit(training_set, training_labels, batch_size=10, epochs=50,
              verbose=1)

# Testing basic model of both sets
test_probs = aps.model.predict(test_set)
test_prediction = test_probs.argmax(axis=1)
rw_probs = aps.model.predict(rw_set)
rw_prediction = rw_probs.argmax(axis=1)

(loss, accuracy) = aps.model.evaluate(test_set, test_labels, verbose=0)
print('\nAccuracy in test set: {}\n'.format(accuracy))
(loss, accuracy) = aps.model.evaluate(rw_set, rw_labels, verbose=0)
print('Accuracy in real world set: {}\n'.format(accuracy))

# Augmenting basic data set to improve performance in real world set
datagen = ImageDataGenerator(rotation_range=10, shear_range=0.3,
                             zoom_range=0.2, width_shift_range=0.15,
                             height_shift_range=0.15, fill_mode='constant',
                             cval=1)

# Creating LeNet with augmmented data
aps_aug = LeNet(training_set[1].shape, conv_1, pool_1, conv_2, pool_2, hidden,
                classes)
aps_aug.model.compile(loss='categorical_crossentropy', optimizer='adam',
                      metrics=["accuracy"])
print('Training LeNet with augmented training data\n')
aps_aug.model.fit_generator(datagen.flow(training_set, training_labels,
                                         batch_size=10),
                            steps_per_epoch=len(training_set), epochs=50,
                            verbose=1)

# Testing augmented model
test_probs_aug = aps_aug.model.predict(test_set)
test_prediction_aug = test_probs_aug.argmax(axis=1)
rw_probs_aug = aps_aug.model.predict(rw_set)
rw_prediction_aug = rw_probs_aug.argmax(axis=1)

(loss, accuracy) = aps_aug.model.evaluate(rw_set, rw_labels, verbose=0)
print('\nAccuracy in real world set: {}'.format(accuracy))
