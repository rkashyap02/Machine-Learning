from scipy.io.wavfile import read
import numpy as np

'''
========================
Linear Predective Coder
========================
NAME:
lpc

SYNOPSIS:
a = lpc(x)

DESCRIPTION:
Program to find the linear predictive parameter vector a for N observations

INPUTS:
x               : Sample set

OUTPUTS:
theta           : Linear Paramters to estimate

AUTHOR:
Rohit Kashyap,2016

'''


def lpc(x):
    """ Given a speech signal of size N = len(x) we predict the next coefficient from all the past values in the 
        windowed frame """
    N = len(x)
    
    # Reverse the array
    x = np.flip(x,0)
    
    ''' Turns out that the sample to estimate is orthogonal to the error from the difference of the sample and the estimate'''
    # Solve the correlation matrix to find the parameters that give the best possible linear estimate
    return np.linalg.solve(np.linalg.pinv(np.matrix(x[0:N-1]).T * np.matrix(x[0:N-1])),(np.matrix(x[0] * x[1:]).T))
    
