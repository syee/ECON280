# -*- coding: utf-8 -*-
"""
Author: Steven Yee
Date: 12/01/2019
Description: Python assignment for ECON 280 Part 1
"""

import numpy
from numpy import genfromtxt
import pandas
from scipy import optimize

#Part 1: Median Regression

sampleSize = 200
replications = 1000
dgBetaEstimates = numpy.zeros(replications)


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
    dgBetaEstimates[count] = estimateBeta(X, Y, sampleSize)

def estimateEmpiricalBeta(X, Y, numObservations):
    return estimateBeta(X, Y, numObservations)

# Is there a better way to read in the data with numpy?
empiricalData = genfromtxt('HW3_data.csv', delimiter=',')
cleanedData = numpy.delete(empiricalData, 0, axis=0)
numObservations = len(cleanedData)
empiricalX = numpy.zeros(numObservations)
empiricalY = numpy.zeros(numObservations)
for count in range(numObservations):
    empiricalX[count] = cleanedData[count][0]
    empiricalY[count] = cleanedData[count][1]

empiricalBetaEstimate = estimateEmpiricalBeta(empiricalX, empiricalY, numObservations)

print("The data generated beta should be approximately 0.6931. This program generated a beta of " + str(numpy.mean(dgBetaEstimates)) + ".")
print("The empirical data should have a beta of approximately 1.3141. This program estimates the beta to be " + str(empiricalBetaEstimate[0]) + ".")
