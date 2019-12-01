# -*- coding: utf-8 -*-
"""
Author: Steven Yee
Date: 12/01/2019
Description: Python assignment for ECON 280
"""

import random
import numpy
import quantecon
import pandas
from scipy import optimize
import matplotlib

#Part 1: Median Regression

sampleSize = 200
replications = 1000
estimates = numpy.zeros(replications)


# Gradient-free optimization method
def optimizationMethod(b, X, Y, length):
    value = (abs(Y - X*b)/length).sum()
    return value


def estimateBeta(X, Y, length):
    estimate = optimize.fmin(optimizationMethod, [1], disp=False, args=(X, Y, length,))
    return estimate


for count in range(replications):
    lamda = 1 + numpy.random.poisson(5, sampleSize)
    X = 1./(lamda + 1)
    Y = numpy.random.exponential(X, sampleSize)
    estimates[count] = estimateBeta(X, Y, sampleSize)

print(numpy.mean(estimates))






