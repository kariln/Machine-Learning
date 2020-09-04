# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 16:55:37 2020

@author: Kari Ness
"""

"""
Logistic regression with gradient descent
"""
import pandas as pd

#Loads 1D dataset
train = pd.read_csv('dataset/regression/train_1d_reg_data.csv')
test = pd.read_csv('dataset/regression/test_1d_reg_data.csv')

class logisticRegression():
    
        #initializing object
        def __init__(self):
            self.weights = None
            
        #extracts the weights from the object
        def getWeights(self):
            return self.weights