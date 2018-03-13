#!/usr/bin/env python
# coding=utf-8
"""
Starter code for logistic regression model to solve OCR task
with MNIST in TensorFlow
MNIST dataset: yann.lecun.com/exdb/mnist/
"""

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import time
# Define paramaters for the model
learning_rate = 0.5
batch_size = 128
n_epochs = 10

mnist = input_data.read_data_sets('/home/allen/Documents/cs20si/stanford-tensorflow-tutorials/examples/data/mnist', one_hot=True)
x = tf.placeholder(shape=[None, 28*28], dtype=tf.float32)
y = tf.placeholder(shape=[None, 10], dtype=tf.uint8)

weight_1 = tf.Variable(tf.random_normal(shape=[28*28, 500], stddev=0.01))
bias_1 = tf.Variable(tf.zeros([500]))

weight_2 = tf.Variable(tf.random_normal(shape=[500, 500], stddev=0.01))
bias_2 = tf.Variable(tf.zeros([500]))

weight_3 = tf.Variable(tf.random_normal(shape=[500, 10], stddev=0.01))
bias_3 = tf.Variable(tf.zeros([10]))

hidden_1 = tf.nn.relu(tf.matmul(x, weight_1) + bias_1)
hidden_2 = tf.nn.relu(tf.matmul(hidden_1, weight_2) + bias_2)
y_out = tf.matmul(hidden_2, weight_3) + bias_3

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_out, labels=y))

train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

start_time = time.time()

n_batches = int(mnist.train.num_examples/batch_size)
for i in range(n_epochs): # train the model n_epochs times
    total_loss = 0

    for _ in range(n_batches):
        x_batch, y_batch = mnist.train.next_batch(batch_size)
        loss_batch, _ = sess.run([loss, train_op], feed_dict={x: x_batch, y: y_batch})
        total_loss += loss_batch
    print('Average loss epoch {0}: {1}'.format(i, total_loss/n_batches))

print('Total time: {0} seconds'.format(time.time() - start_time))
print('Optimization Finished!') # should be around 0.35 after 25 epochs

# test the model
preds = tf.nn.softmax(y_out)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(preds, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
print('Accuracy {0:7.3f}%'.format(acc))
