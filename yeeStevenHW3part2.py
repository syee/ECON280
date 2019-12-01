# -*- coding: utf-8 -*-
"""
Author: Steven Yee
Date: 12/01/2019
Description: Python assignment for ECON 280
"""

import numpy
from scipy.stats.distributions import chi2
from scipy.stats import norm

#Part 2: Bootstrapping the Median

sampleSize = 100
bootstrapReplications = 500
monteCarloReplications = 10000

trueMedian = chi2.ppf(0.5, 3)
z95 = norm.ppf(0.95)

C_asy = numpy.zeros(monteCarloReplications)
C_boot = numpy.zeros(monteCarloReplications)

for count in range(monteCarloReplications):
    X = chi2.rvs(3, size = bootstrapReplications)
    sampleMedian = numpy.median(X)
    sampleVariance = 1/(4*bootstrapReplications*((chi2.pdf(sampleMedian, 3))**2))
    CI_low = sampleMedian - z95*numpy.sqrt(sampleVariance)
    CI_high = sampleMedian + z95*numpy.sqrt(sampleVariance)
    C_asy[count] = (trueMedian > CI_low) and (trueMedian < CI_high)

    #Bootstrap
    medianStar = numpy.zeros(bootstrapReplications)
    for i in range(bootstrapReplications):
        X_star = numpy.random.choice(X, sampleSize)
        medianStar[i] = numpy.median(X_star)

    CI_low_b = numpy.quantile(medianStar, 0.05)
    CI_high_b = numpy.quantile(medianStar, 0.95)
    C_boot[count] = (trueMedian > CI_low_b) and (trueMedian < CI_high_b)

Asy_coverage = numpy.mean(C_asy)
Boot_coverage = numpy.mean(C_boot)

print("Asy coverage is " + str(Asy_coverage))
print("Bootstrap coverage is " + str(Boot_coverage))