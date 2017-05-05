# QML-RG@ICFO Homework 1: Autoencoder
# Alejandro Pozas-Kerstjens

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

n_input = 10      # Number of inputs
n_coded = 6       # Number of nodes in encoder

# Adam parameters
learning_rate = 0.001
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8

ins = tf.placeholder("float", shape=(n_input,None))      # Initialize inputs as variables. None denotes arbitrary value

encode = tf.Variable(tf.zeros([n_coded,n_input]))    # Initialize encoder and decoder matrices
decode = tf.Variable(tf.zeros([n_input,n_coded]))

outs_true = ins
encoded = tf.nn.sigmoid(tf.matmul(encode,ins))
outs_pred = tf.nn.sigmoid(tf.matmul(decode,encoded))

# Define cost function (mean square error) and optimization procedure
cost = tf.reduce_mean(tf.pow(outs_true - outs_pred, 2))
optimizer = tf.train.AdamOptimizer(learning_rate,beta1,beta2,epsilon).minimize(cost)

# Tools to evaluate the model
correct_pred = tf.equal(tf.argmax(outs_true, 1),tf.argmax(outs_pred, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

#Initialization of training procedure
init = tf.global_variables_initializer()

nsteps=100000
sess = tf.Session()     # Define session
sess.run(init)      # Initialize session (begin computing stuff)

feedins = np.random.choice([0,1],size=(n_input,100))
for i in range(nsteps):
    sess.run(optimizer, feed_dict = {ins: feedins})      # Train
    print(sess.run(accuracy, feed_dict = {ins: feedins}))
    
# One-instance test
test = np.random.choice([0,1],size=(n_input,1))
encode_decode_test = sess.run(outs_pred, feed_dict={ins: test})
plt.plot(range(n_input),test,'ro',range(n_input),encode_decode_test,'b^')
plt.show()