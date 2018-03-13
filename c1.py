#!/usr/bin/env python
# coding=utf-8
import tensorflow as tf 
#x = 2
#y = 3
#op1 = tf.add(x,y)
#op2 = tf.add(x,y)
#op3 = tf.multiply(op1,op2)
#useless = tf.pow(op1,op2)
#with tf.Session() as sess:
#    Op3,Useless = sess.run([op3,useless])

#with tf.device('/cpu:0'):
#    a = tf.constant([1.0,2.0,3.0,4.0,5.0,6.0],name = 'a')
#    b = tf.constant([1.0,2.0,3.0,4.0,5.0,6.0],name = 'b')
#    c = tf.multiply(a,b)

#sess = tf.Session(config = tf.ConfigProto(log_device_placement = True))

#print sess.run(c)
#print Op3 
#print Useless 

g = tf.Graph()
with g.as_default():
    a = 3
    b = 5
    x = tf.add(a,b)
sess = tf.Session(graph = g)
print sess.run(x)

sess.close()

