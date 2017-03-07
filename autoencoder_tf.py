import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

n_input = 100
n_coded = 50

encode = tf.Variable(tf.random_uniform([c_coded,n_input]))
decode = tf.Variable(tf.random_uniform([c_coded,n_input]))

vec_in = tf.placeholder("float32", shape=(n_input,None))

vec_encoded = tf.sigmoid(tf.matmul(encode,vec_in))
vec_out = tf.sigmoid(tf.matmul(decode,vec_encoded))

cost = 