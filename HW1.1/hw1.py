#!/usr/bin/env python3
import json 
import pandas as pd 
import statistics
import matplotlib
import matplotlib.pyplot as plt
from random import randrange
from sklearn.model_selection import train_test_split
import numpy as np
from scipy.optimize import minimize

class Data:
    def __init__(self,address,is_adult = 0):
        self.flag = is_adult
        self.data = pd.read_json(address)
        self.mean_x = np.mean(self.data['x'])
        self.mean_y = np.mean(self.data['y'])
        self.std_x = np.std(self.data['x'])
        self.std_y = np.std(self.data['y'])
        if is_adult == 1 :
            self.data = self.data[self.data['x'] < 18]
            self.mean_x = np.mean(self.data['x'])
            self.mean_y = np.mean(self.data['y'])
            self.std_x = np.std(self.data['x'])
            self.std_y = np.std(self.data['y'])

    def normalization(self):
        self.data['x'] = [(i-self.mean_x)/self.std_x for i in self.data['x']]
        self.data['y'] = [(i -self.mean_y)/self.std_y for i in self.data['y']]
    
    def visualization(self):
        if self.flag == 'classification':
            plt.scatter(self.data['y'],self.data['is_adult'])
            plt.show()
        else:
            plt.scatter(self.data['x'],self.data['y'])
            plt.show()

    def train_test_split(self,split):
        if self.flag != 'classification':
            x_train, x_test, y_train, y_test = train_test_split(self.data['x'], self.data['y'], test_size=split, random_state=1)
            return x_train,x_test,y_train,y_test
        else: 
            x_train, x_test, y_train, y_test = train_test_split(self.data['y'], self.data['is_adult'], test_size=split, random_state=1)
            return x_train,x_test,y_train,y_test


class LinearRegression:
    def __init__(self,x,y):
        self.x = x 
        self.y = y 
        self.w = [0,0]


    def loss(self,p):
        result = np.dot(self.x,p[0]) + p[1]
        e = self.y - result 
        se = np.power(e,2)
        rse = np.sqrt(np.sum(se))
        rmse = rse/self.y.shape[0]
        return rmse 

    def optimize(self):
        result = minimize(self.loss,self.w,method = 'nelder-mead')
        return result.x
    
# Linear Regression 
data = Data('weight.json',1)
data.visualization()
x_train, x_test, y_train, y_test = data.train_test_split(0.2)
lr = LinearRegression(x_train,y_train)
coefficient = lr.optimize()
#plot fitting curve 
plt.plot(x_train,y_train,'o')
plt.plot(x_train,x_train*coefficient[0] + coefficient[1])
plt.show()
data.normalization()
x_train, x_test, y_train, y_test = data.train_test_split(0.2)
y_pred = x_test*coefficient[0] + coefficient[1]



# class LogisticRegression:
#     def __init__(self, learning_rate=0.001, n_iters=10000):
#         self.lr = learning_rate
#         self.n_iters = n_iters
#         self.weights = None
#         self.bias = None

#     def fit(self, X, y):
#         n_samples = X.shape[0]
#         # init parameters
#         self.weights = 0
#         self.bias = 0


#         # gradient descent
#         for _ in range(self.n_iters):
#             # approximate y with linear combination of weights and x, plus bias
#             linear_model = np.dot(X, self.weights) + self.bias
#             # apply sigmoid function
#             y_predicted = self._sigmoid(linear_model)

#             # compute gradients
#             dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
#             db = (1 / n_samples) * np.sum(y_predicted - y)
#             # update parameters
#             self.weights -= self.lr * dw
#             self.bias -= self.lr * db

#     def predict(self, X):
#         linear_model = np.dot(X, self.weights) + self.bias
#         y_predicted = self._sigmoid(linear_model)
#         y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
#         return np.array(y_predicted_cls)

#     def _sigmoid(self, x):
#         return 1 / (1 + np.exp(-x))




class LogisticRegression:
    def __init__(self,x,y):
        self.x = x 
        self.y = y
        self.mean_x = np.mean(self.x)
        self.mean_y = np.mean(self.y)
        self.std_x = np.std(self.x)
        self.std_y = np.std(self.y)
    
    def pred(self,para,x):
        A = para[0]
        w = para[1]
        x0 = para[2]
        s = para[3] 
        y_pred = A/(1+np.e**((x0-x)/w)) + s
        return y_pred

    def loss_function(self,para):
        y_pred = self.pred(para,self.x)
        error = self.y - y_pred 
        se = np.power(error,2)
        rse = np.sqrt(np.sum(se))
        rmse = rse / self.y.shape[0] 
        return rmse 
        

    def optimize(self):
        result = minimize(self.loss_function,[0,0,0,0],method = 'Powell')
        return result.x


# Logistic Regression fitting the original graph 
data_2 = Data('weight.json','logistic')
data_2.visualization()
x_train, x_test, y_train, y_test = data_2.train_test_split(0.2)
lg2 = LogisticRegression(x_train,y_train)
y_predicted = lg2.pred(lg2.optimize(),x_train)
plt.plot(x_train,y_train,'o')
plt.scatter(x_train,y_predicted, c = 'red')
plt.show()
    



# Logistic Regression for classification case
data_1 = Dat·a('weight.json','classification')
data_1.visualization()
x_train, x_test, y_train, y_test = data_1.train_test_split(0.2)

lg = LogisticRegression(x_train,y_train)
y_predicted = lg.pred(lg.optimize(),x_train)
plt.plot(x_train,y_train,'o')
plt.scatter(x_train,y_predicted,c = 'red') 
plt.show()



        

        


















