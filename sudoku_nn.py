from input_data import load_data
import numpy as np
import os

SAMPLE_SIZE = 100000 #could range from 1 to 1,000,000

X, Y = load_data(SAMPLE_SIZE)

import tensorflow as tf
#placeholder to hold data while training
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
x = tf.placeholder("float32", shape=(9, 9), name="input_data")
y = tf.placeholder("float32", shape=(9, 9), name="output_data")

#initialize weights with random values
W_input_hidden = tf.Variable(tf.random_uniform(shape=(9, 9), minval= -10.0,maxval=10.0), name="W_in-h")
W_hidden_output = tf.Variable(tf.random_uniform(shape=(9, 9), minval= -10.0,maxval=10.0), name="W_h-ot")

tf.summary.histogram("Weights_input-hidden",  W_input_hidden)
tf.summary.histogram("Weights_hidden-output", W_hidden_output)

with tf.name_scope("Wx") as scope:
    hidden = tf.nn.softmax(tf.matmul(x,W_input_hidden))
    model = tf.nn.elu(tf.matmul(hidden,W_hidden_output))

with tf.name_scope("cost_function") as scope:
    cost_function = tf.reduce_mean(tf.square(y - model))
    tf.summary.scalar("cost_function", cost_function)

with tf.name_scope("train") as scope:
    optimizer = tf.train.RMSPropOptimizer(0.2).minimize(cost_function)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.5)
    sess.run(init)
    average_cost = 0.0
    summary_writer = tf.summary.FileWriter('/home/sudoku/logs', graph=sess.graph)
    for index in range(len(x_train)):
        x_ = x_train[index]
        y_ = y_train[index]
        x_ = x_.reshape(9,9)
        y_ = y_.reshape(9,9)
        sess.run(optimizer, feed_dict={x: x_, y: y_})
        cost = sess.run(cost_function, feed_dict={x: x_, y: y_})
        if (index + 1) % 100 == 0:
            print "iteration_cost : ", cost
            merged = tf.summary.merge_all()
            summary_str = sess.run(merged, feed_dict={x: x_, y: y_})
            summary_writer.add_summary(summary_str)
        x_ = sess.run(model, feed_dict={x: x_})
        sess.run(optimizer, feed_dict={x: x_, y: y_})
