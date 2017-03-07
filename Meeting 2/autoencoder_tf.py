# QML-RG@ICFO Homework 1: Autoencoder

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

n_input = 100
n_coded = 10

encode = tf.Variable(tf.random_uniform([n_coded,n_input])) #variables to learn
decode = tf.Variable(tf.random_uniform([n_input,n_coded]))

vec_in = tf.placeholder("float32", shape=(n_input,None)) #placeholder that could be vector or matrix

vec_encoded = tf.sigmoid(tf.matmul(encode,vec_in))  #matrix multiplication to get reduced vector
vec_out = tf.sigmoid(tf.matmul(decode,vec_encoded)) #matrix multiplication to return to initial size

cost = tf.reduce_mean(tf.square(vec_out - vec_in)) #least squared error
optimizer = tf.train.AdamOptimizer()  #choosing optimizer method
train = optimizer.minimize(cost)      #training by minimizing cost function

nsteps = 1000
init = tf.global_variables_initializer() #learning variables must be initialized
sess = tf.Session() #start graph session
sess.run(init)

#training network to reproduce polynomials up to third order
x = np.linspace(-1,1,n_input)
for i in range(1000):
    

#training
for i in range(nsteps):
    sess.run(train, {vec_in: training_data})
    print(sess.run(cost,{vec_in: training_data}))

test_input = np.random.uniform(size = [n_input,1])
test_output = sess.run(vec_out,{vec_in: test_input})
plt.plot(range(n_input),test_input,'ro',range(n_input),test_output,'b^')
plt.show()