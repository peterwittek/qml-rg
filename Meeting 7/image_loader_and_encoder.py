import math
import os
from skimage import io
from skimage import transform as tf
from matplotlib import pyplot as plt
from skimage.transform import resize
import numpy as np
"""This code loads the images from the APS capture and encodes them with a neural network
in a lower dimensional vector.
For 20 hidden layers and 1000 epochs the encoding is already quite good. (This can be seen by
comparing the original and the encoded-decoded image)
To get to even lower dimensions, we manually lower the dimension in the end.
"""
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
        X[i]=resize(X[i],(32,32,1))#reduce the size of the image from 100X100 to 32X32. Also flattens the color levels
    X=[np.reshape(x, (1024,)) for x in X] #reshape list (1024,) is crucial
    Y = xlabels
    return X,Y

#------------------------------------------------------------------------------
#Load and reshape Images with image_loader.py
training_set, training_labels= load_images("images/train/")
test_set, test_labels = load_images("images/test/")

resize_set, resize_labels= prep_datas(training_set, training_labels)
resize_test_set, resize_test_labels= prep_datas(test_set, test_labels)
#print(resize_set)
#---------------------------------------------------------------
#very ugly way to bring vectors to the right shape for SVC fit()
a = []
for x in resize_set:
    a.append(x.tolist())
X = a

from keras.models import Model, Sequential
from keras.layers import Input, Dense
from keras import regularizers

#----------------------------------------------------------------
#Nice code from Keras example
encoding_dim = 100

input_img = Input(shape=(1024,))
# add a Dense layer with a L1 activity regularizer
encoded = Dense(encoding_dim, activation='relu')(input_img)
decoded = Dense(1024, activation='sigmoid')(encoded)

#----------------------------------------------------------------
#this model paps the input to the reconstruction
autoencoder = Model(input_img, decoded)
#----------------------------------------------------------------
# this model maps an input to its encoded representation
encoder = Model(input_img, encoded)

#----------------------------------------------------------------
#And the decoder
# create a placeholder for an encoded (n-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))
#-------------------------------------------------------------------
#This is now the real training
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
autoencoder.fit(X, X, epochs=1000, batch_size=10)

autoencoder.save("save_model.h5") #save model

#--------------------------------------------------------------------
#To use the model in the end we need these two functions
# encode and decode some digits
# note that we take them from the *test* set

encoded_imgs = encoder.predict(X)
decoded_imgs = decoder.predict(encoded_imgs)

#---------------------------------------------------------------------
#NOW compare resized (32x32) image with encoded image
decoded_imgs = [np.reshape(x,  (32,32)) for x in decoded_imgs] #reshape vectors again to images
original_imgs = [np.reshape(x, (32,32)) for x in X]           #reshape vectors again to images

for i in range(0,len(X)):
    decoded =   np.asarray(decoded_imgs[i])
    original = np.asarray(original_imgs[i])
    plt.subplot(2,1,1)
    plt.imshow(decoded.squeeze(), cmap = 'gray')
    plt.subplot(2,1,2)
    plt.imshow(original.squeeze(), cmap = 'gray')  
    plt.show()

#------------------------------------------------------------------------
#The vector encoded_imgs now is the data we need for the SVD!
#encoding_dim can be changed, depending on how many input dimensions we
#want for the SVD

#We can also reduce the dimensions again manually
outputdim = 4 #what final size do we want
length = len(encoded_imgs) #how many pictures
output = []

for k in range(0,length):
    encoded = encoded_imgs[k] #which picture do we want to compress?
    list = []
    for i in range(0, outputdim):
        list.append(sum([x for x in encoded[i:(i+int(encoding_dim/outputdim))]]))
    output.append(list)
print(output)



