#!/usr/bin/env python
# coding=utf-8
import tensorflow as tf
a = tf.constant(3)
b = tf.constant(5)
x = tf.add(a,b)
with tf.Session() as sess:
    print sess.run(x)
writer = tf.summary.FileWriter('./log',sess.graph)
writer.close()
