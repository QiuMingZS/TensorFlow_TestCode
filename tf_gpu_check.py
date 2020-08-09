# -*- coding: UTF-8 -*-
import tensorflow as tf

with tf.device('/cpu:0'):
    a = tf.constant([1.0, 5.0], shape=[2], name='a')
    b = tf.constant([2.0, 2.0], shape=[2], name='b')
with tf.device('/gpu:0'):
    c = a + b

sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(c))
