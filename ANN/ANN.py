#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

#  展示25张8*8大小的手写数据集
def show_digits():
    digists = datasets.load_digits()
    fig = plt.figure()
    for i in range(25):
        ax = fig.add_subplot(5, 5, i + 1)
        ax.imshow(digists.images[i], cmap='gray_r')
    plt.show()


# 构建神经网络
# 输入参数分别为输入数据，输入数据特征个数，输出个数，是否使用激活函数
def add_layer(inputs, in_size, out_size, activation_function=None):
    # 设置权重和偏置
    weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    # 计算xw+b
    xw_plus_b = tf.matmul(inputs, weights) + biases
    # 是否使用激活函数
    if activation_function is None:
        outputs = xw_plus_b
    else:
        outputs = activation_function(xw_plus_b)
    return outputs


# 加载数据
digists = datasets.load_digits()
# 数据为numpy格式
x_data = digists.data
y_data = digists.target.reshape(-1, 1)
# 将x_data 缩放到[0, 1]之间
x_data = MinMaxScaler().fit_transform(x_data)
# 将标签y_data one-hot编码，y_data变为十列
y_data = OneHotEncoder().fit_transform(y_data).todense()
n_input = x_data.shape[1]   # 特征个数
n_class = y_data.shape[1]   # 数字类别个数
# 将80%数据集作为训练集，20%作为测试集
x_trian, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=1)
# 定义占位符
x_tf = tf.placeholder(tf.float32, [None, n_input])
y_tf = tf.placeholder(tf.float32, [None, n_class])
# 添加第一个隐含层，令隐含层节点数out_size=128，使用激活函数
l1 = add_layer(x_tf, n_input, 128, activation_function=tf.nn.relu)
# 添加第二个隐含层，令隐含层节点数out_size=256，使用激活函数
l2 = add_layer(l1, 128, 256, activation_function=tf.nn.relu)
# 添加输出层, 不使用激活函数
pred = add_layer(l2, 256, n_class, activation_function=None)
# 定义损失函数,采用交叉熵
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y_tf))
# 使用优化器最小化损失
train_step = tf.train.AdamOptimizer(learning_rate=0.05).minimize(loss)
# 计算准确率
y_pred = tf.argmax(pred, 1)
bool_pred = tf.equal(tf.argmax(y_tf, 1), y_pred)
accuracy = tf.reduce_mean(tf.cast(bool_pred, tf.float32))
# 初始化变量
init = tf.global_variables_initializer()

# 构建会话
with tf.Session() as sess:
    sess.run(init)
    print('训练集迭代结果')
    for step in range(100):
        sess.run(train_step, feed_dict={x_tf: x_trian, y_tf: y_train})
        # 每迭代十次输出一次
        if step % 10 == 0:
            # 误差损失
            LOSS = sess.run(loss, feed_dict={x_tf: x_trian, y_tf: y_train})
            # 准确率
            res = sess.run(accuracy, feed_dict={x_tf: x_data, y_tf: y_data})
            print('train: {} | loss: {} | accuracy: {}'.format(step, LOSS, res))
    print('测试集结果')
    print('Accurcay:',  accuracy.eval({x_tf: x_test, y_tf: y_test}))
