# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 14:57:57 2020

@author: Мамусик
"""
import numpy as np
X = np.array([[1,2,3],[4,5,6], [7,8,9], [2,1,0]])
W = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
y = np.array([0,1,2,0])
num_train = X.shape[0]
num_classes = W.shape[1]

dW = np.zeros(W.shape) 
loss = 0.0

scores = np.dot(X, W)
pred_scores = np.argmax(scores, axis = 1)
