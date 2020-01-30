# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 14:01:14 2020

@author: Эличка
"""
import numpy as np
X_train = np.array([[20,41,12],[1,22,3],[2,3,4],[3,4,5]])
X_test = np.array([[0,1,2],[1,2,35]])
y_train = np.array([0, 1, 2, 1])
num_test = X_test.shape[0]
num_train = X_train.shape[0]
dists = np.zeros((num_test, num_train))


'''
# two loops
for i in range(num_test):
    for  j in range(num_train):
        dists[i,j] = np.sqrt(np.sum(np.square((X_train[j] - X_test[i]))))
print(dists)

# one loop
for i in range(num_test):
    dists[i,:] =np.sqrt(np.sum(np.square(X_train - X_test[i, :]), axis = 1))
print(dists)
'''
# no loops
dists = np.sqrt(np.sum(X_train**2, axis =1) + np.reshape(np.sum(X_test**2, axis=1), [num_test, 1])
- 2*np.dot(X_test, X_train.T))


y_pred = np.zeros(num_test)

for i in range(num_test):
    closest_y = []
    ind = np.argsort(dists[i])
    closest_y = y_train[ind][0:3]
    k = np.bincount(closest_y)
    y_pred[i] = k.argmax()