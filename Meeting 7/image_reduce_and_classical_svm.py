# -*- coding: utf-8 -*-
"""Created on Wed Mar 15 10:09:24 2017"""
import os
import numpy as np
from skimage import io
from matplotlib import pyplot as plt
from skimage.transform import resize
from sklearn import svm, metrics
from sklearn.decomposition import PCA
from keras.models import Model
from keras.layers import Input, Dense


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


def prep_images(images, size):
    for i in range(len(images)):
        images[i] = resize(images[i], size)
        images[i] = np.reshape(images[i], size[0]*size[1])
        if np.amax(images[i]) != 0:
            images[i] = images[i]/np.amax(images[i])
    return np.array(images)


# Loading image data sets
training_set, training_labels = load_images("images/train/")
test_set, test_labels = load_images("images/test/")
rw_set, rw_file_labels = load_images("images/real_world/")
training_orig = training_set[:]
test_orig = test_set[:]
rw_orig = rw_set[:]
# Resize images to same dimensions, normalize the grey scale and return vectors
image_size = (100, 100)
training_set = prep_images(training_set, image_size)
test_set = prep_images(test_set, image_size)
rw_set = prep_images(rw_set, image_size)
# Sort real world set by number in file name
rw_set = np.array([x for (y, x) in sorted(zip(rw_file_labels, rw_set))])
# Get labels for real world set from file
f = open('images/real_world/labels.txt', "r")
lines = f.readlines()
rw_labels = []
for x in lines:
    rw_labels.append(int((x.split('	')[1]).replace('\n', '')))
f.close()
# Reduce labels to only two categories
training_labels = np.array(training_labels)
training_labels[training_labels > 0] = 1
rw_labels = np.array(rw_labels)
rw_labels[rw_labels > 0] = 1

# Now compress images in various ways for input into SVM
# First method: skimage transform resize
compress_size = (2, 2)
training_resize = prep_images(training_orig, compress_size)
test_resize = prep_images(test_orig, compress_size)

# Second method: PCA
pca = PCA(n_components=compress_size[0]*compress_size[1])
training_pca = pca.fit_transform(training_set)
test_pca = pca.transform(test_set)
pca_norm = np.amax(training_pca)
training_pca = training_pca/pca_norm  # normalizing for better SVM
test_pca = test_pca/pca_norm

# Third method: autoencoder
input_dim = image_size[0]*image_size[1]
encoding_dim = compress_size[0]*compress_size[1]
input_img = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu')(input_img)
decoded = Dense(input_dim, activation='sigmoid')(encoded)
autoencoder = Model(input_img, decoded)
encoder = Model(input_img, encoded)
encoded_input = Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

autoencoder.fit(training_set, training_set,
                epochs=2000, batch_size=10, verbose=0)

training_auto = encoder.predict(training_set)
test_auto = encoder.predict(test_set)
auto_norm = np.amax(training_auto)
training_auto = training_auto/auto_norm  # normalizing for better SVM
test_auto = test_auto/auto_norm

# Support vector machine with resized images
svm_resize = svm.SVC(kernel='rbf', C=5000)
svm_resize.fit(training_resize, training_labels)

svm_resize_predict = svm_resize.predict(test_resize)

print("\nSVM with resized images\n")
print("Training set confusion matrix:\n%s"
      % metrics.confusion_matrix(training_labels,
                                 svm_resize.predict(training_resize)))
print("Test set confusion matrix:\n%s"
      % metrics.confusion_matrix(test_labels,
                                 svm_resize_predict))

# SVM with data reduced by PCA
svm_pca = svm.SVC(kernel='rbf', C=5000)
svm_pca.fit(training_pca, training_labels)

svm_pca_predict = svm_pca.predict(test_pca)

print("\nSVM with images reduced by PCA\n")
print("Test set confusion matrix:\n%s"
      % metrics.confusion_matrix(training_labels,
                                 svm_pca.predict(training_pca)))
print("Test set confusion matrix:\n%s"
      % metrics.confusion_matrix(test_labels,
                                 svm_pca_predict))

# SVM with data reduced by autoencoder
svm_auto = svm.SVC(kernel='rbf', C=10000)
svm_auto.fit(training_auto, training_labels)

svm_auto_predict = svm_auto.predict(test_auto)

print("\nSVM with images reduced by autoencoder\n")
print("Test set confusion matrix:\n%s"
      % metrics.confusion_matrix(training_labels,
                                 svm_auto.predict(training_auto)))
print("Test set confusion matrix:\n%s"
      % metrics.confusion_matrix(test_labels,
                                 svm_auto_predict))
