from __future__ import print_function
import numpy as np
import os

SAMPLE_SIZE = 100000 #could range from 1 to 1,000,000

print("Loading dataset...")
from input_data import load_data
X, Y = load_data(SAMPLE_SIZE)
print("done.")

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#placeholder to hold data while training
x = tf.placeholder("float32", shape=(9, 9), name="input_data")
y = tf.placeholder("float32", shape=(9, 9), name="output_data")

#initialize weights with random values
W_input_hidden = tf.Variable(tf.random_uniform(shape=(9, 6), minval= -10.0,maxval=10.0), name="W_in-h")
W_hidden_hidden = tf.Variable(tf.random_uniform(shape=(6, 6), minval= -10.0,maxval=10.0), name="W_h-h")
W_hidden_output = tf.Variable(tf.random_uniform(shape=(6, 9), minval= -10.0,maxval=10.0), name="W_h-ot")

tf.summary.histogram("Weights_input-hidden",  W_input_hidden)
tf.summary.histogram("Weights_hidden-output", W_hidden_hidden)
tf.summary.histogram("Weights_hidden-output", W_hidden_output)

with tf.name_scope("Wx") as scope:
    hidden = tf.nn.relu6(tf.matmul(x,W_input_hidden))
    hidden1 = tf.nn.relu6(tf.matmul(x,hidden))
    model = tf.nn.relu6(tf.matmul(hidden1,W_hidden_output))

#cost function seems in effective TODO: write a better cost function
with tf.name_scope("cost_function") as scope:
    cost_function = tf.reduce_mean(tf.square(y - model))
    tf.summary.scalar("cost_function", cost_function)

with tf.name_scope("train") as scope:
    optimizer = tf.train.AdagradOptimizer(learning_rate=0.05).minimize(cost_function)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.5)
    sess.run(init)
    summary_writer = tf.summary.FileWriter('/home/sudoku/logs', graph=sess.graph)
    print("Begin training...")
    for index in range(len(x_train)):
        x_ = x_train[index]
        y_ = y_train[index]
        x_ = x_.reshape(9,9)
        y_ = y_.reshape(9,9)
        sess.run(optimizer, feed_dict={x: x_, y: y_})
        if (index + 1) % 100 == 0:
            cost = sess.run(cost_function, feed_dict={x: x_, y: y_})
            print("\rCOST : ",cost,end="")
            merged = tf.summary.merge_all()
            summary_str = sess.run(merged, feed_dict={x: x_, y: y_})
            summary_writer.add_summary(summary_str)
    print("\nTraining successful.")
    print("Starting testing.")
    average_error = 0.0
    for index in range(len(x_test)):
        x_ = x_test[index]
        y_ = y_test[index]
        x_ = x_.reshape(9,9)
        y_ = y_.reshape(9,9)
        y_out = sess.run(model, feed_dict={x: x_})
        if (index + 1) % 100 == 0:
            error = tf.reduce_mean(tf.square(y_ - y_out))
            error = error.eval()
            average_error += error
            print("\rERROR : ",error,end="")
    print("\nTesting finished.")
    print("Error: %f"%(average_error/((len(x_test))/100)))
