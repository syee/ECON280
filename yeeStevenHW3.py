# -*- coding: utf-8 -*-
"""
Author: Steven Yee
Date: 12/01/2019
Description: Python assignment for ECON 280
"""

import random
import numpy
from numpy import genfromtxt
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

def estimateEmpiricalBeta(X, Y, numObservations):
    return estimateBeta(X, Y, numObservations)

# Is there a better way to read in the data?
empiricalData = genfromtxt('HW3_data.csv', delimiter=',')
cleanedData = numpy.delete(empiricalData, 0, axis=0)
numObservations = len(cleanedData)
empiricalX = numpy.zeros(numObservations)
empiricalY = numpy.zeros(numObservations)
for count in range(numObservations):
    empiricalX[count] = cleanedData[count][0]
    empiricalY[count] = cleanedData[count][1]

empiricalEstimate = estimateEmpiricalBeta(empiricalX, empiricalY, numObservations)

print("The data generated beta should be approximately 0.6931. This program generated a beta of " + str(numpy.mean(estimates)) + ".")
print("The empirical data should have a beta of approximately 1.3141. This program estimates the beta to be " + str(empiricalEstimate[0]) + ".")
