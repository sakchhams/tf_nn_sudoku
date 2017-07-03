from __future__ import print_function
import numpy as np
import os

#Parameters
LOG_DIR = '/home/tf_nn_sudoku/logs'
learning_rate = 0.001
training_iters = 200000 #could range from 1 to 1,000,000
batch_size = 200
display_step = 4
#N-Network Parameters
n_input = 81 # sudoku puzzle : 9*9 matrix
n_output = 81 # sudoku puzzle (solved) : 9*9 matrix
dropout = 0.75 # dropout, probability to keep units

print("Loading dataset...")
from input_data import load_data
sudoku_dataset = load_data(training_iters)
print("done.")

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#placeholder to hold data while training
x = tf.placeholder("float32", [batch_size, n_input], name="input_data")
y = tf.placeholder("int32", [batch_size, n_output], name="output_data")
keep_probability = tf.placeholder(tf.float32)

#creates a convolutional layer
def Convolution2D(x, W, b, strides=1):
    #conv2d parameters:
    #x : 4d tensor for which 2d convolution is computed, tensor of the shape [batch, in_height, in_width, in_channels]
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu6(x)

#MaxPool for convolutional layers
def MaxPool2D(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

#create a model
def create_conv_network(x, weights, biases, dropout):
    #reshape the input tensor as 4 dimensional
    x = tf.reshape(x, shape=[-1, 9, 9, 1])

    #Layer1
    l1 = Convolution2D(x, weights['wl1'], biases['bl1'])
    l1 = MaxPool2D(l1, k=2) #down sampling
    print("l1.shape: ", l1.shape)

    #Layer2
    l2 = Convolution2D(l1, weights['wl2'], biases['bl2'])
    l2 = MaxPool2D(l2, k=2) #down sampling
    print("l2.shape: ", l2.shape)

    #Layer3 - fully connected layer
    fcl = tf.reshape(l2, shape=[-1, 3*3*64])
    fcl = tf.add(tf.matmul(fcl, weights['wd1']), biases['bd1'])
    fcl = tf.nn.relu6(fcl)
    fcl = tf.nn.dropout(fcl, dropout) #apply dropout
    print("fcl-dim:",fcl.shape)

    return tf.add(tf.matmul(fcl, weights['out']), biases['out'])

#initialize weights
weights = {
    #5x5 convolution 1 input 32 outputs
    'wl1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    #5x5 convolution 32 inputs, from previous layer, 64 outputs
    'wl2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    #fully connected, 3*3*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([3*3*64, 1024])),
    #1024 inputs, 81 outputs
    'out': tf.Variable(tf.random_uniform([1024, n_output]))
}

#initialize biases
biases = {
    'bl1': tf.Variable(tf.random_normal([32])),
    'bl2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_uniform([n_output]))
}

prediction_model = create_conv_network(x, weights, biases, keep_probability)
print("prediction_model.shape :", prediction_model.shape)
cost_function = tf.reduce_mean(tf.square(prediction_model - tf.to_float(y)))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost_function)

correct_pred = tf.equal(tf.argmax(prediction_model, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    epoch = 1
    while epoch * batch_size < training_iters:
        batch_x, batch_y = sudoku_dataset.train.next_batch(batch_size)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_probability: dropout})
        if epoch % display_step == 0:
            loss, acc = sess.run([cost_function, accuracy], feed_dict={x: batch_x, y: batch_y, keep_probability: 1.})
            print("Iter " + str(epoch*batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
        epoch += 1
    print("Optimization Finished!")
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: sudoku_dataset.test.problems[:256], y: sudoku_dataset.test.solutions[:256], keep_probability: 1.}))
