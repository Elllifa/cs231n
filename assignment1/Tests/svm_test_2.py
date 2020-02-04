# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 14:57:39 2020

@author: Эличка
"""
import numpy as np
X = np.array([[1,2,3],[4,5,6], [7,8,9], [2,1,0]])
W = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
y = np.array([0,1,2,0])
num_train = X.shape[0]
num_classes = W.shape[1]

dW = np.zeros(W.shape) 
loss = 0.0

scores = X.dot(W)

correct_class_scores =  np.choose(y, scores.T)
margin = scores - correct_class_scores.reshape(scores.shape[0], 1) + 1
loss+= np.sum(margin[margin>0]) - y.shape[0]

margin[np.arange(num_train), y] =0