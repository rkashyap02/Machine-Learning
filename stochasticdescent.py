'''
=======================
Stochastic Gradient Descent
=======================
NAME:
stochasticdescent

SYNOPSIS:
theta = stochasticdescent(x,y)

DESCRIPTION:
Program to find the linear parameter theta using Stochastic descent

INPUTS:
x               : Input training set
y               : Output to the input training values

OUTPUTS:
theta           : Paramter to estimate

AUTHOR:
Rohit Kashyap,2016

'''

import numpy as np
import matplotlib.pyplot as plt

def stochasticdescent(x,y):
    # Compute the pseudo inverse of (x.T * x) as the Matrix may not be invertible
    theta = np.linalg.pinv(x.T * x) * (x.T * y)
    return theta
