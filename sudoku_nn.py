#!/usr/bin/python
from __future__ import print_function
import numpy as np
import os

#Parameters
LOG_DIR = '/home/sakchham/machinel/tf_nn_sudoku/logs'
MODEL_STORE = '/home/sakchham/machinel/tf_nn_sudoku/data/sudoku_model.ckpt'
learning_rate = 0.001
training_iters = 200000 #could range from 1 to 1,000,000
testing_iters = 200000
batch_size = 200
display_step = 2
#N-Network Parameters
n_input = 81 # sudoku puzzle : 9*9 matrix
n_output = 81 # sudoku puzzle (solved) : 9*9 matrix
n_blocks = 10

print("Loading dataset...")
from input_data import load_data
sudoku_dataset = load_data(training_iters+testing_iters)
print("done.")

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#placeholder to hold data while training
x = tf.placeholder("float32", name="input_data")
y = tf.placeholder("int32", name="output_data")

#creates a convolutional layer
def Convolution2D(inputs, filters=None, kernel_size=1):
    outputs = tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=kernel_size, padding="same", activation=tf.nn.relu)
    return batch_normalize(outputs)

def batch_normalize(tensor):
    return tf.contrib.layers.batch_norm(inputs=tensor,
                                       decay=0.99,
                                       center=True,
                                       scale=True,
                                       activation_fn=tf.nn.relu,
                                       updates_collections=None,
                                       zero_debias_moving_mean=True,
                                       fused=True)

#create a model
def create_conv_network(x):
    #reshape the input tensor as 4 dimensional
    x_enc = tf.reshape(x, shape=[-1, 9, 9, 1])
    for _ in range(n_blocks):
        x_enc = Convolution2D(inputs=x_enc, filters=512, kernel_size=3)

    logits = Convolution2D(inputs=x_enc, filters=10, kernel_size=1)
    return logits

istarget = tf.to_float(tf.equal(x, tf.zeros_like(x)))
logits = create_conv_network(x)
probabilities = tf.reduce_max(tf.nn.softmax(logits), axis=-1)
predictions = tf.to_int32(tf.arg_max(logits, dimension=-1))
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
cost_function = tf.reduce_sum(cross_entropy * istarget) / (tf.reduce_sum(istarget))
tf.summary.scalar("cost_function", cost_function)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost_function)

hits = tf.to_float(tf.equal(predictions, y)) * istarget
accuracy = tf.reduce_sum(hits) / (tf.reduce_sum(istarget) + 1e-8)

init = tf.global_variables_initializer()

def train(sess):
    epoch = 1
    while epoch * batch_size < training_iters:
        batch_x, batch_y = sudoku_dataset.train.next_batch(batch_size)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if epoch % display_step == 0:
            loss, acc = sess.run([cost_function, accuracy], feed_dict={x: batch_x, y: batch_y})
            print("\rIter " + str(epoch*batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc), end="")
        epoch += 1
    print("\nOptimization Finished!")

with tf.Session() as sess:
    sess.run(init)
    saver = tf.train.Saver()
    from os.path import isfile
    if not isfile(MODEL_STORE+".index"):
        print("Training:")
        train(sess)
        save_path = saver.save(sess, MODEL_STORE)
    else:
        print("Restoring session:")
        saver.restore(sess, MODEL_STORE)
    x_test, y_test = sudoku_dataset.test.next_batch(200)
    pred = sess.run(predictions, feed_dict={x: x_test, y: y_test})
    print("x", x_test[0])
    print("predictions", pred[0])
    print("y", y_test[0])
