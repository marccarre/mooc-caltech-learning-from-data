from __future__ import division
import sys
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
import random

print(sys.version + '\n')

d = 2 # dimensionality of the problem
N = 1000 # number of training samples
percentageToNoisify = 10 # percentage of training samples to force to be wrong
N_test = 100 # number of testing samples
a = -1.0 # lower bound for X
b = 1.0 # upper bound for X
numSimulations = 1000

def sign(x):
    return 1.0 if (x > 0) else -1.0

def label(X, x1, x2, y1, y2):
    xp = (x2 - x1) * (y2 - X[1]) - (y2 - y1) * (x2 - X[0])
    return sign(xp)

def f(x1, x2):
    return sign(x1**2 + x2**2 - 0.6)

def noisify(y, N, percentageToNoisify):
    numToNoisify = int(N * percentageToNoisify / 100)
    indexesToNoisify = random.sample(range(N), numToNoisify)
    for i in indexesToNoisify:
        y[i] = -1 * y[i]
    return y

def runSimulations(numSimulations, N, percentageToNoisify):
    E_ins = []
    E_outs = []

    for _ in xrange(numSimulations):

        # Generate arbitrary boundary:
        x1, y1 = (b - a) * np.random.rand(d) + a
        x2, y2 = (b - a) * np.random.rand(d) + a

        # Generate data-set:
        X = np.column_stack((np.ones((N, 1)), (b - a) * np.random.rand(N, d) + a))
        y_raw = np.array([label(X[j], x1, x2, y1, y2) for j in xrange(N)])
        y = noisify(y_raw, N, percentageToNoisify)

        # Train using closed-form linear regression:
        X_dagger = np.dot(linalg.inv(np.dot(X.T, X)), X.T)
        w = np.dot(X_dagger, y)

        # Estimate target hypothesis and calculate in-sample error:
        g = np.vectorize(sign)(np.dot(X, w))
        E_in = np.average(y - g)
        E_ins += [E_in]

    return (E_ins, E_outs)

print('Started linear regression simulation...')
E_ins, E_outs = runSimulations(numSimulations, N, percentageToNoisify)
E_in_avg = np.average(E_ins)
E_out_avg = np.average(E_outs)
print('In-sample error:     %0.5f' % E_in_avg)
print('Out-of-sample error: %0.5f' % E_out_avg)

