#!/usr/bin/env python
# -*- coding: utf-8 -*-
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# 定义权重
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)
# 定义偏置
def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)
# 定义卷积层
def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')
# 定义池化层
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 加载数据
mnist = input_data.read_data_sets("../mnist", one_hot=True)
# 定义占位符
x_tf = tf.placeholder(tf.float32, [None, 784])
x_image = tf.reshape(x_tf, [-1, 28, 28, 1])
y_tf = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

# 第一层卷积
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
# 第二层卷积
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
# 第三层 全连接层
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)  # dropout
# 第四层 Softmax输出层
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# 模型训练及评估
cross_entropy = -tf.reduce_sum(y_tf * tf.log(y_conv))  # 计算交叉熵
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)  # 使用adam优化器来以0.001的学习率来进行迭代
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_tf, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # 准确率

# 初始化参数
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for step in range(1000):
        x_batch, y_batch = mnist.train.next_batch(100)
        train_step.run(feed_dict={x_tf: x_batch, y_tf: y_batch, keep_prob: 0.5})
        if step % 100 == 0:
            res = sess.run(accuracy, feed_dict={x_tf: x_batch, y_tf: y_batch, keep_prob: 1.0})
            print('train: {} | train accuracy: {}'.format(step, res))
    print("test accuracy %g" % accuracy.eval({x_tf: mnist.test.images, y_tf: mnist.test.labels, keep_prob: 1.0}))
