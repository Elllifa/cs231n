#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 21:58:04 2020

@author: pelv
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
scores = X.dot(W)
correct_class_scores =  np.choose(y, scores.T).reshape(scores.shape[0],1)
margin = scores - correct_class_scores+1   
print(margin, margin.shape)
margin = np.maximum(0, margin)
print(margin)
# subtract y.shape[0] because later we added 1 to all axes, and now we
# have to return all margins to zeros for the correct classes:
loss+= np.sum(margin[margin>0]) - y.shape[0] 

loss /= num_train
#loss += reg * np.sum(W * W) 
margin[np.arange(num_train), y] = 0 
# count how much positions need to be modified
margin[margin>0] = 1
samples_to_subtract = np.sum(margin, axis = 1)
# for every margin>0 we must subtract one sample of x from the correct class:
margin[range(num_train), y]-=samples_to_subtract 
# now we have matrix that shows the count of rows of X that we must add
# to every column of dW
dW =  np.dot(X.T, margin)

#dW += reg * 2 * W
                                                #
