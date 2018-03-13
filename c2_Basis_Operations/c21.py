#!/usr/bin/env python
# coding=utf-8:w
import tensorflow as tf
a = tf.constant(2,name = 'a')
b = tf.constant(3,name = 'b')
x = tf.add(a,b,name = 'add')
with tf.Session() as sess:
    print sess.run(x)
writer = tf.summary.FileWriter('./log',sess.graph)
writer.close()
