# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 13:07:26 2020

@author: Kari Ness
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#reads the data from the csv file with pandas
train = pd.read_csv('dataset/regression/train_1d_reg_data.csv')
test = pd.read_csv('dataset/regression/test_1d_reg_data.csv')

#splits the datasets into x and y
x_train = train.drop(columns = "y");
x_train["bias"] = 1; #appends the bias
y_train = train["y"];

x_test = test.drop(columns = "y");
x_test["bias"] = 1; #appends the bias
y_test = test["y"];


#creating a class to make linear regression models
class linearRegression():
        #initializing object
        def __init__(self):
            self.weights = None
        
        #extracts the weights from the object
        def getWeights(self):
            return self.weights
        
        #calculates and sets weights on object. Trains the model.
        def setWeights(self,X,Y):
            inverse = np.linalg.inv(X.T.dot(X))
            self.weights = (inverse.dot(X.T)).dot(Y);
        
        #calculates Y with weights and X
        def predictor(self,X):
            return np.dot(np.array([self.weights]),(X.T))

        
        #calculates the error with ordinary least square, OLS
        def calc_E (self,X,Y,weights):
            return (((X.np.dot(weights)-Y)**2)).np.sum()/(len(Y)) # Denne vil gi feil!

#train new model:
model = linearRegression()
model.setWeights(x_train,y_train)
[weight_x1, bias] = model.getWeights() # Fjernet weight_x1 ettersom [...]_1d:reg.csv har kun 1-dim. 


#Predicting the train and test data
y_train_pred = model.predictor(x_train)
print(y_train_pred.shape)
print(x_train["x1"].shape)
y_test_pred = model.predictor(x_test)

# Plots the results
fig = plt.figure(figsize=(20,15))
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)
ax1.set_title("Training Predictions",fontsize = 30)
ax1.set_xlabel("Input",fontsize=25)
ax1.set_ylabel("Target",fontsize=25)
ax1.scatter(x_train["x1"],y_train,c='b')
ax1.plot(x_train["x1"],y_train_pred[0], linestyle = 'solid', color = 'r')
ax1.legend(["Model","Target data"], prop={'size': 25})

ax2.set_title("Testing Predictions")
ax2.set_xlabel("Input",fontsize=25)
ax2.set_ylabel("Target",fontsize=25)
ax2.scatter(x_test["x1"],y_test,c='b')
ax2.plot(x_test["x1"],y_test_pred[0], c='r')
plt.show()