#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 14 12:59:45 2018

source: http://cs231n.github.io/neural-networks-case-study/
"""
import numpy as np
import matplotlib.pyplot as plt


plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

X = np.genfromtxt("x.txt", delimiter=" ")
y = np.loadtxt("y.txt")
fig = plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.xlim([-1,1])
plt.ylim([-1,1])
#fig.savefig('spiral_data.png')

# train a linear classifier
import tensorflow as tf
import tflearn
from tflearn.data_utils import to_categorical

with tf.Graph().as_default():
    
    net = tflearn.input_data([None, 2])
    net = tflearn.fully_connected(net, 100, activation='relu')
    net = tflearn.fully_connected(net, 3, activation='softmax')
    gd = tflearn.SGD(learning_rate=1.0)
    net = tflearn.regression(net, optimizer=gd, loss='categorical_crossentropy')

    Y = to_categorical(y,3)
    lm = tflearn.DNN(net,tensorboard_verbose=0)
    lm.fit(X, Y, show_metric=True, batch_size=len(X), n_epoch=1000, snapshot_epoch=False)
    
    
    
# plot the resulting classifier
h = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = np.argmax(lm.predict(np.c_[xx.ravel(), yy.ravel()]), axis=1)
Z = Z.reshape(xx.shape)
fig = plt.figure()
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.show()
#fig.savefig('spiral_net.png')
print("Accuracy: {}%".format(100 * np.mean(y == np.argmax(lm.predict(X), axis=1))))
