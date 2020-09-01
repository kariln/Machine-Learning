# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 15:01:04 2020

@author: Kari Ness
"""

"""
Object-oriented version of linear regression
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#reads the data from the csv file with pandas
train = pd.read_csv('dataset/regression/train_2d_reg_data.csv')
test = pd.read_csv('dataset/regression/test_2d_reg_data.csv')

#splits the datasets into x and y
x_train = train.drop(column = "y");
x_train["bias"] = 1; #appends the bias
y_train = train["y"];

x_test = test.drop(column = "y");
x_test["bias"] = 1; #appends the bias
y_test = test["y"];


#creating a class to make linear regression models
class linearRegression:
        #initializing object
        def __init__(self):
            self.weights = None
        
        #extracts the weights from the object
        def getWeights(self):
            return self.weights
        
        #calculates and sets weights on object. Trains the model.
        def setWeights(self,X,Y):
            self.weights = np.dot(np.inv(np.dot(X.T,X)),np.dot(X.T,Y));
        
        #calculates Y with weights and X
        def predictor(self,X):
            return np.array(np.dot(self.weights,X))
        
        #calculates the error with ordinary least square, OLS
        def calc_E (self,X,Y,weights):
            return (((X.np.dot(weights)-Y)**2)).np.sum()/(len(Y))

    