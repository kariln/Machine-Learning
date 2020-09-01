# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 14:48:52 2020

@author: Kari Ness
"""

"""
Linear regression with ordinary least squares (OLS)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#reads the data from the csv file with pandas and converts the Pandas DataFrame to a Numpy Array
data_train2 = pd.read_csv('dataset/regression/train_2d_reg_data.csv').to_numpy()
data_test2 = pd.read_csv('dataset/regression/test_2d_reg_data.csv').to_numpy()

x1_train = data_train2[:,1];
x2_train = data_train2[:,2];
y_train = data_train2[:,2];

N = len(x1_train);

x1_test = data_test2[:,1];
x2_test = data_test2[:,2];
y_test = data_test2[:,2];


#Finds weights in model1
w1 = np.dot(x1_train.T,y_train)/(np.dot(x1_train.T,x1_train));

#Finds error in model1
E1_train = 1/N*(abs(np.dot(x1_train,w1)-y_train)**2).np.sum()
#print(E1_train)
E1_test = 1/N*(abs(np.multiply(x1_test,w1)-y_test)**2).np.sum()
#print(E1_test)

#Regression line
h = np.multiply(w1.T,x1_train)
#print(h)

#Plotting the training set with the regression line
fig = plt.figure()

# Ploting Line
plt.plot(x1_train, h, color='#58b970', label='Regression Line')
# Ploting Scatter Points
plt.scatter(x1_train, y_train, c='#ef5423', label='Scatter Plot')
        
plt.title('Linear regression')
plt.xlabel('X')
plt.ylabel('Y')
fig.savefig('linear_regression.png', dpi=fig.dpi)