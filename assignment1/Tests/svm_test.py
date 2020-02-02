# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 11:42:14 2020

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
sc=[]
for i in range(num_train):
    scores = np.dot(X[i], W)
    sc.append(scores)
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
        if j == y[i]:
            continue
        margin = scores[j] - correct_class_score + 1 # note delta = 1
        if margin > 0:
            loss += margin
            dW[:,j] = dW[:,j]+X[i]
            dW[:, y[i]] = dW[:, y[i]]-X[i]