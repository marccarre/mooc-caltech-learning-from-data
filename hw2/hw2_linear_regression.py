from __future__ import division
import sys
import numpy as np
from numpy import linalg, sign
import matplotlib.pyplot as plt

print(sys.version + '\n')

d = 2 # dimensionality of the problem
N = 100 # number of training samples
N_test = 1000 # number of testing samples
a = -1.0 # lower bound for X
b = 1.0 # upper bound for X
numSimulations = 1000

def f(X, p1, p2):
    z = np.cross(p2-p1, p2-X)
    return sign(z[0])
    
def plot(X, y, p1, p2):
    plt.axis([-1.1, 1.1, -1.1, 1.1])
    plt.plot([p1[1],p2[1]], [p1[2],p2[2]],color='r')
    pos = np.array([X[i] for i in xrange(N) if y[i] == 1.0])
    neg = np.array([X[i] for i in xrange(N) if y[i] == -1.0])
    plt.plot(pos[:,1], pos[:,2], 'o')
    plt.plot(neg[:,1], neg[:,2], 'x')
    plt.show()

def runSimulations(numSimulations, N, N_test):
    E_ins = []
    E_outs = []
    for i in xrange(numSimulations):

        # Generate arbitrary boundary:
        p1 = np.concatenate([np.array([1]), (b - a) * np.random.rand(d) + a])
        p2 = np.concatenate([np.array([1]), (b - a) * np.random.rand(d) + a])

        # Generate data-set:
        X = np.column_stack((np.ones((N, 1)), (b - a) * np.random.rand(N, d) + a))
        y = np.array([f(X[j,:], p1, p2) for j in xrange(N)])

        # Train using closed-form linear regression:
        X_dagger = np.dot(linalg.inv(np.dot(X.T, X)), X.T)
        w = np.dot(X_dagger, y)

        # Estimate target hypothesis and calculate in-sample error:
        g = np.vectorize(sign)(np.dot(X, w))
        E_in = np.average(sign(abs(g-y)))
        E_ins += [E_in]

        # Generate testing data-set:
        X_test = np.column_stack((np.ones((N_test, 1)), (b - a) * np.random.rand(N_test, d) + a))
        y_test = np.array([f(X_test[j,:], p1, p2) for j in xrange(N_test)])

        # Apply target hypothese and calculate out-of-sample error:
        g_test = np.vectorize(sign)(np.dot(X_test,w))
        E_out = np.average(sign(abs(g_test-y_test)))
        E_outs += [E_out]

        if (i == (numSimulations - 1)):
            # Apply PLA after having calculated w using linear regression (which should speed up convergence):
            perceptronLearningAlgorithm(1000, 10, p1, p2, w)
            
            # Plot classification boundary and points:
            plot(X, y, p1, p2)

    return (E_ins, E_outs)

def perceptronLearningAlgorithm(numSimulations, N, p1, p2, w = None, maxIter = 10000):
    print('Started PLA (%i simulations)...' % numSimulations)
    iters = []
    for _ in xrange(numSimulations):
        # Generate data-set:
        X = np.column_stack((np.ones((N, 1)), (b - a) * np.random.rand(N, d) + a))
        y = np.array([f(X[j,:], p1, p2) for j in xrange(N)])

        hasMisclassifiedSample = True
        iter = 0
        while (hasMisclassifiedSample and (iter < maxIter)):
            hasMisclassifiedSample = False
            for i in xrange(N):
                if (y[i] != sign(np.dot(X[i], w))):
                    hasMisclassifiedSample = True
                    w = w + y[i] * X[i]
            iter += 1
        iters += [iter]
    print('PLA converged in %0.5f iteration(s) (min=%i, max=%i)' % (np.average(iters), np.min(iters), np.max(iters)))
    return iters

print('Started Linear Regression (%i simulations)...' % numSimulations)
E_ins, E_outs = runSimulations(numSimulations, N, N_test)
print('In-sample error:     avg=%0.5f, min=%0.5f, max=%0.5f, median=%0.5f' % (np.average(E_ins), np.min(E_ins), np.max(E_ins), np.median(E_ins)))
print('Out-of-sample error: avg=%0.5f, min=%0.5f, max=%0.5f, median=%0.5f' % (np.average(E_outs), np.min(E_outs), np.max(E_outs), np.median(E_outs)))

