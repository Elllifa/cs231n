# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 19:07:03 2020

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

num_train = X.shape[0]
scores = np.dot(X,W)
for i in range(scores.shape[0]):
    scores[i] -= np.max(scores[i])        
    numen = scores[i][y[i]]
    nemun =  np.exp(scores[i][y[i]])
    softmax = np.exp(scores[i][y[i]]) / np.sum(np.exp(scores[i]))
    loss += -np.log(softmax)
    
    for j in range(scores.shape[1]):
        dW[:,j] += np.exp(scores[i][j]) / np.sum(np.exp(scores[i]))*X[i]
    dW[:,y[i]] -= X[i]
loss /= num_train
dW /= num_train
