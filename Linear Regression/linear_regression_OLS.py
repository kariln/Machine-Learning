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
w1 = np.multiply(np.multiply(x1_train.T,y_train),(np.multiply(x1_train.T,x1_train)));
print(len(w1))
print(len(x1_test))
print(N)

#Finds error in model1
E1_train = 1/N*abs(np.multiply(x1_train,w1)-y_train)**2
#E1_test = 1/N*abs(np.multiply(x1_test,w1)-y_test)**2 #må sørge for like størrelse