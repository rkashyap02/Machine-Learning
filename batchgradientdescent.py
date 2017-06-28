'''
=======================
Batch Gradient Descent
=======================
NAME:
batchgradientdescent

SYNOPSIS:
theta = batchgradientdescent(x,y,alpha,theta,numIterations)

DESCRIPTION:
Program to find the linear parameter theta using Gradient descent

INPUTS:
x               : Input training set
y               : Output to the input training values
alpha           : Learning rate
theta           : Paramter to estimate
numIterations   : Total iterations to hope J(theta) converges

OUTPUTS:
theta           : Paramter to estimate

AUTHOR:
Rohit Kashyap,2016

'''

import numpy as np
import matplotlib.pyplot as plt
import random

def  batchgradientdescent(x,y,alpha,theta,numIterations):
    for i in range(0,numIterations):
        # Compute the error from the hypothesis
        error = x * theta - y
        # Find the steepest slope or gradient
        gradient = x.T * error / len(x)
        # Estimate theta
        theta = theta - alpha * gradient
        return theta
    

