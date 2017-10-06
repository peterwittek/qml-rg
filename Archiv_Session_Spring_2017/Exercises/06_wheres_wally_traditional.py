# -*- coding: utf-8 -*-
"""Created on Wed Mar 15 10:09:24 2017"""
import math
import os
import numpy as np
from skimage import io
from skimage import transform as tf
from matplotlib import pyplot as plt
from skimage.transform import resize
from sklearn import svm, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import xgboost as xgb
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


def prep_images(images,size):
    for i in range(len(images)):
        images[i] = resize(images[i], size)
        images[i] = np.reshape(images[i], size[0]*size[1])/np.amax(images[i])
    return np.array(images)


image_size = (32, 32)

# Loading image data sets
training_set, training_labels = load_images("images/train/")
test_set, test_labels = load_images("images/test/")
rw_set, rw_file_labels = load_images("images/real_world/")
# Resize images to same dimensions, normalize the grey scale and return vectors
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
# Support vector machine
svm_model = svm.SVC(kernel='linear', C=10)
svm_model.fit(training_set, training_labels)

svm_test_predict = svm_model.predict(test_set)
svm_rw_predict = svm_model.predict(rw_set)

print("\nSVM\n")
print("Test set confusion matrix:\n%s" % metrics.confusion_matrix(test_labels, svm_test_predict))
print("Real world confusion matrix:\n%s" % metrics.confusion_matrix(rw_labels, svm_rw_predict))

# Random forest
forest_model = RandomForestClassifier(n_estimators=100)
forest_model.fit(training_set, training_labels)

rf_test_predict = forest_model.predict(test_set)
rf_rw_predict = forest_model.predict(rw_set)

print("\nRandom forest\n")
print("Test set confusion matrix:\n%s" % metrics.confusion_matrix(test_labels, rf_test_predict))
print("Real world confusion matrix:\n%s" % metrics.confusion_matrix(rw_labels, rf_rw_predict))

# XGBoost
xgb_model = xgb.XGBClassifier(scale_pos_weight=23/2)
xgb_model.fit(training_set, training_labels)

xgb_test_predict = xgb_model.predict(test_set)
xgb_rw_predict = xgb_model.predict(rw_set)

print("\nXGBoost\n")
print("Test set confusion matrix:\n%s" % metrics.confusion_matrix(test_labels, xgb_test_predict))
print("Real world confusion matrix:\n%s" % metrics.confusion_matrix(rw_labels, xgb_rw_predict))

# Reducing dimension of data to two principal components for visualization
pca = PCA(n_components=2)
train_pca = pca.fit_transform(training_set)

# create a mesh to plot in
h = .1  # step size in the mesh
x_min, x_max = train_pca[:, 0].min() - 1, train_pca[:, 0].max() + 1
y_min, y_max = train_pca[:, 1].min() - 1, train_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

for i, clf in enumerate((svm_model, forest_model, xgb_model)):
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    plt.subplot(1, 3, i + 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    Z = clf.predict(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()]))

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

    # Plot also the training points
    plt.scatter(train_pca[:, 0], train_pca[:, 1], c=training_labels,
                cmap=plt.cm.coolwarm)
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

plt.show()

input_dim = image_size[0]*image_size[1]
encoding_dim = 2

# Reducing dimension of data using autoencoder
input_img = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu')(input_img)
decoded = Dense(input_dim, activation='sigmoid')(encoded)
autoencoder = Model(input_img, decoded)
encoder = Model(input_img, encoded)
encoded_input = Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.fit(training_set, training_set,
                epochs=2000, batch_size=25, shuffle=True, verbose=0)

encoded_imgs = encoder.predict(training_set)

# create a mesh to plot in
h = .1  # step size in the mesh
x_min, x_max = encoded_imgs[:, 0].min(), encoded_imgs[:, 0].max()
y_min, y_max = encoded_imgs[:, 1].min(), encoded_imgs[:, 1].max()
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

for i, clf in enumerate((svm_model, forest_model, xgb_model)):
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    plt.subplot(1, 3, i + 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    Z = clf.predict(decoder.predict(np.c_[xx.ravel(), yy.ravel()]))

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

    # Plot also the training points
    plt.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1], c=training_labels,
                cmap=plt.cm.coolwarm)
    plt.xlabel('Auto 1')
    plt.ylabel('Auto 2')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

plt.show()
