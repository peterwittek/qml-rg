# Author: Gorka Muñoz

# Example of use of the confusion scheme introduced in https://arxiv.org/abs/1610.02048 
# to differenciate phases. In this case, we will try to differentiate between two numbers
# of the CIFAR database. The value we get from the Linear Discriminant Analysis will make
# the form of the order parameter. The goal is to find the critical point, e.g. in which
# value of the order parameter there is a phase transition. In the context of the CIFAR,
# we want to find the value of the LDA which differentiates between the two chosen numbers. 

import numpy as np
import matplotlib.pyplot as plt
from sklearn import discriminant_analysis
from keras.datasets import mnist
from scipy.signal import argrelextrema
from keras.models import Sequential
from keras.layers import Dense


#-----------------------------------------------------------------------------
# Data preparation

(x_train, y_train), (x_test, y_test) = mnist.load_data()

number_1 = 7
number_2 = 0

## Train set
x_train = x_train[(y_train == number_1) | (y_train == number_2)]
y_train = y_train[(y_train == number_1) | (y_train == number_2)]

X_m = x_train.astype(float)
y_m = y_train.astype(float)

X_m = X_m.reshape(X_m.shape[0], X_m.shape[1]**2)

X_m = X_m[1:5000,:] # we truncate the data to have a faster program. Feel free to test with hole database
y_m = y_m[1:5000]

## Test set
x_test = x_test[(y_test == number_1) | (y_test == number_2)]
y_test = y_test[(y_test == number_1) | (y_test == number_2)]

x_test = x_test.astype(float)
y_test = y_test.astype(float)

x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]**2)



#-----------------------------------------------------------------------------
# Projection on to the first 2 linear discriminant components

print("Computing Linear Discriminant Analysis projection")
# Train
X2 = X_m.copy()
X2.flat[::X_m.shape[1] + 1] += 0.01  # Make X invertible
X_lda = discriminant_analysis.LinearDiscriminantAnalysis(n_components=2).fit_transform(X2, y_m)

plt.figure()
plt.title('Linear Discriminant Analysis')
plt.scatter(X_lda[(y_m == number_1)], y_m[y_m == number_1]) 
plt.scatter(X_lda[(y_m == number_2)], y_m[y_m == number_2]) 

# Test
X3 = x_test.copy()
X3.flat[::x_test.shape[1] + 1] += 0.01  # Make X invertible
X_lda_test = discriminant_analysis.LinearDiscriminantAnalysis(n_components=2).fit_transform(X3, y_test)



#-----------------------------------------------------------------------------
# Neural Network

model = Sequential()
model.add(Dense(80, input_dim=X_m.shape[1], init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


#-----------------------------------------------------------------------------
# Confusion scheme

# We go through
values_c = np.linspace(min(X_lda)[0], max(X_lda)[0], 15)

accuracy = []
for c in values_c:
    
    print(np.where(values_c == c)[0][0])
    y_nn = np.zeros_like(y_m)
    y_nn[(X_lda[:,0] < c)] = 0 
    y_nn[(X_lda[:,0] > c)] = 1
         
    model.reset_states
    model.fit(X_m, y_nn, epochs=5, batch_size=10,  verbose=1)
    
    predictions = model.predict(x_test)
    
    y_nn_test = np.zeros_like(y_test)
    y_nn_test[(X_lda_test[:,0] < c)] = 0 
    y_nn_test[(X_lda_test[:,0] > c)] = 1
    
    accuracy.append(sum(abs(predictions[:,0] - y_nn_test)))
    

# Results

critical_value = values_c[argrelextrema(np.array(accuracy), np.less)[0][0]]

plt.figure()
plt.title('W_shape')
plt.xlabel('LDA value')
plt.xlabel('Accuracy of the NN')
plt.plot(values_c, accuracy)




plt.figure()
plt.title('Linear Discrimant Analysis')
plt.ylabel('Number of CIFAR')
plt.xlabel('LDA value')
plt.scatter(X_lda_test, y_test,)
plt.plot((critical_value, critical_value), (number_1, number_2), 'r-', label = 'Critical point')
plt.legend()











