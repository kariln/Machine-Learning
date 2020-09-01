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
            self.weights = np.dot(np.linalg.inv(np.dot(X.T,X)),np.dot(X.T,Y));
        
        #calculates Y with weights and X
        def predictor(self,X):
            return np.array(np.dot(self.weights,X.T))
        
        #calculates the error with ordinary least square, OLS
        def calc_E (self,X,Y,weights):
            return (((X.np.dot(weights)-Y)**2)).np.sum()/(len(Y))

#train new model:
model = linearRegression()
model.setWeights(x_train,y_train)

#Plotting the training set with the regression line
train_x1 = plt.figure()
y_train = model.predictor(x_train)
y_test = model.predictor(x_test)

# Ploting Line
plt.plot(x1, y1, color='#58b970', label='Model')
# Ploting Scatter Points
plt.scatter(x1, y1, c='#ef5423', label='Target data')

#formatting the plot
plt.title('Linear regression')
plt.xlabel('X')
plt.ylabel('Y')
leg = plt.legend()
train_x1.savefig('linear_regression.png', dpi=train.x1.dpi)
    