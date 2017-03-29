# QML-RG@ICFO Homework 1: variant of autoencoder to recreate low order polynomials

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

n_input = 40 #number of input data points
n_coded = 4  #number of coded data points

#adam parameters
learning_rate = 0.01
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8

#variables to learn
encode = tf.Variable(tf.random_normal([n_coded,n_input])) 
decode = tf.Variable(tf.random_normal([n_input,n_coded]))

#placeholder input to learn from that could be vector or matrix
vec_in = tf.placeholder("float32", shape=(n_input,None)) 

#coding and decoding transformations, no need for activation functions with intended data set
vec_encoded = tf.matmul(encode,vec_in)  
vec_out = tf.matmul(decode,vec_encoded) 

#optimization choice
cost = tf.reduce_mean(tf.square(vec_out - vec_in)) #least squared error
optimizer = tf.train.AdamOptimizer(learning_rate,beta1,beta2,epsilon)  #choosing optimizer method
train = optimizer.minimize(cost)      #training by minimizing cost function

#initilizing for training                          
nsteps = 10000
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

#training data set of random third order polynomials
x = np.linspace(-1,1,n_input).reshape([n_input,1])
coeffs = np.random.uniform(size = [4, 10000], low = -1.0)
training_data = coeffs[0,:] + coeffs[1,:]*x + coeffs[2,:]*x**2 + coeffs[3,:]*x**3   
    
#training
for i in range(nsteps):
    sess.run(train, {vec_in: training_data})
    print(sess.run(cost,{vec_in: training_data}))

#plot result for random test data
tcoeff = np.random.uniform(size = [4], low = -1.0)
test_input = tcoeff[0] + tcoeff[1]*x + tcoeff[2]*x**2 + tcoeff[3]*x**3   
test_output = sess.run(vec_out,{vec_in: test_input})
plt.plot(range(n_input),test_input,'ro',range(n_input),test_output,'b^')
plt.show()

#close session
sess.close()