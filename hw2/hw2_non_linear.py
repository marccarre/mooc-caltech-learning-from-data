from __future__ import division
import sys
import numpy as np
from numpy import linalg, sign
import matplotlib.pyplot as plt
import random

print(sys.version + '\n')

d = 2 # dimensionality of the problem
N = 1000 # number of training samples
N_test = 1000 # number of testing samples
a = -1.0 # lower bound for X
b = 1.0 # upper bound for X
numSimulations = 1000
percentageToNoisify = 10 # percentage of training samples to force to be wrong

def f(X):
    return sign(X[1]**2 + X[2]**2 - 0.6)

def noisify(y, N, percentageToNoisify):
    numToNoisify = int(N * percentageToNoisify / 100)
    indexesToNoisify = random.sample(range(N), numToNoisify)
    for i in indexesToNoisify:
        y[i] = -1 * y[i]
    return y

def runLinear(numSimulations, N, percentageToNoisify):
    E_ins = []
    
    for _ in xrange(numSimulations):
        # Generate data-set:
        X = np.column_stack((np.ones((N, 1)), (b - a) * np.random.rand(N, d) + a))
        y = noisify(np.array([f(X[j,:]) for j in xrange(N)]), N, percentageToNoisify)

        # Train using closed-form linear regression:
        X_dagger = np.dot(linalg.inv(np.dot(X.T, X)), X.T)
        w = np.dot(X_dagger, y)

        # Estimate target hypothesis and calculate in-sample error:
        g = np.vectorize(sign)(np.dot(X, w))
        E_in = np.average(sign(abs(g-y)))
        E_ins += [E_in]

    return E_ins

def runNonLinear(numSimulations, N, N_test, percentageToNoisify):
    E_ins = []
    E_outs = []
    weights = []
    
    for i in xrange(numSimulations):

        # Generate data-set:
        X_linear = np.column_stack((np.ones((N, 1)), (b - a) * np.random.rand(N, d) + a))
        X = np.column_stack((X_linear, X_linear[:,1]*X_linear[:,2], X_linear[:,1]**2, X_linear[:,2]**2))
        y = noisify(np.array([f(X_linear[j,:]) for j in xrange(N)]), N, percentageToNoisify)

        # Train using closed-form linear regression:
        X_dagger = np.dot(linalg.inv(np.dot(X.T, X)), X.T)
        w = np.dot(X_dagger, y)

        # Estimate target hypothesis and calculate in-sample error:
        g = np.vectorize(sign)(np.dot(X, w))
        E_in = np.average(sign(abs(g-y)))
        E_ins += [E_in]

        # Generate testing data-set:
        X_test_linear = np.column_stack((np.ones((N_test, 1)), (b - a) * np.random.rand(N_test, d) + a))
        X_test = np.column_stack((X_test_linear, X_test_linear[:,1]*X_test_linear[:,2], X_test_linear[:,1]**2, X_test_linear[:,2]**2))
        y_test = noisify(np.array([f(X_test_linear[j,:]) for j in xrange(N_test)]), N_test, percentageToNoisify)

        # Apply target hypothese and calculate out-of-sample error:
        g_test = np.vectorize(sign)(np.dot(X_test,w))
        E_out = np.average(sign(abs(g_test-y_test)))
        E_outs += [E_out]

        weights += [w]

    return (E_ins, E_outs, weights)

print('Started Linear Regression (%i simulations)...' % numSimulations)
E_ins = runLinear(numSimulations, N, percentageToNoisify)
print('In-sample error:     avg=%0.5f, min=%0.5f, max=%0.5f, median=%0.5f\n' % (np.average(E_ins), np.min(E_ins), np.max(E_ins), np.median(E_ins)))

print('Started Non-Linear Regression (%i simulations)...' % numSimulations)
E_ins, E_outs, weights = runNonLinear(numSimulations, N, N_test, percentageToNoisify)
print('In-sample error:     avg=%0.5f, min=%0.5f, max=%0.5f, median=%0.5f' % (np.average(E_ins), np.min(E_ins), np.max(E_ins), np.median(E_ins)))
print('Out-of-sample error: avg=%0.5f, min=%0.5f, max=%0.5f, median=%0.5f' % (np.average(E_outs), np.min(E_outs), np.max(E_outs), np.median(E_outs)))
print('Weights: %s' % np.average(weights, axis=0))

