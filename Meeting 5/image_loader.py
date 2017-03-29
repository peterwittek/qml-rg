# -*- coding: utf-8 -*-
"""Created on Wed Mar 15 10:09:24 2017"""
import math
import os
from skimage import io
from skimage import transform as tf


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


def deshear(filename):
    image = io.imread(filename)
    distortion = image.shape[1] - image.shape[0]
    shear = tf.AffineTransform(shear=math.atan(distortion/image.shape[0]))
    return tf.warp(image, shear)[:, distortion:]

training_set, training_labels = load_images("images/train/")
test_set, test_labels = load_images("images/test/")
