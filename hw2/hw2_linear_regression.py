from __future__ import division
import sys
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt

print(sys.version + '\n')

d = 2 # dimensionality of the problem
N = 100 # number of training samples
N_test = 1000 # number of testing samples
a = -1.0 # lower bound for X
b = 1.0 # upper bound for X
numSimulations = 1000

def sign(xp):
    return 1.0 if (xp > 0) else -1.0

def label(X, x1, x2, y1, y2):
    xp = (x2 - x1) * (y2 - X[1]) - (y2 - y1) * (x2 - X[0])
    return sign(xp)
    
def plot(X, y, x1, x2, y1, y2):
    plt.axis([-1.1, 1.1, -1.1, 1.1])
    plt.plot([x1,x2], [y1,y2],color='r')
    pos = np.array([X[i] for i in xrange(N) if y[i] == 1.0])
    neg = np.array([X[i] for i in xrange(N) if y[i] == -1.0])
    plt.plot(pos[:,0], pos[:,1], 'o')
    plt.plot(neg[:,0], neg[:,1], 'x')
    plt.show()

def runSimulations(numSimulations, N, N_test):
    E_ins = []
    E_outs = []
    for i in xrange(numSimulations):

        # Generate arbitrary boundary:
        x1, y1 = (b - a) * np.random.rand(d) + a
        x2, y2 = (b - a) * np.random.rand(d) + a

        # Generate data-set:
        X = (b - a) * np.random.rand(N, d) + a
        y = np.array([label(X[j], x1, x2, y1, y2) for j in xrange(N)])

        # Train using closed-form linear regression:
        X_dagger = np.dot(linalg.inv(np.dot(X.T, X)), X.T)
        w = np.dot(X_dagger, y)

        # Estimate target hypothesis and calculate in-sample error:
        g = np.vectorize(sign)(np.dot(X, w))
        E_in = np.average(y - g)
        E_ins += [E_in]

        # Generate testing data-set:
        X_test = (b - a) * np.random.rand(N_test, d) + a
        y_test = np.array([label(X_test[j], x1, x2, y1, y2) for j in xrange(N_test)])

        # Apply target hypothese and calculate out-of-sample error:
        g_test = np.vectorize(sign)(np.dot(X_test,w))
        E_out = np.average(y_test - g_test)
        E_outs += [E_out]

        if (i == (numSimulations - 1)):
            # Apply PLA after having calculated w using linear regression (which should speed up convergence):
            perceptronLearningAlgorithm(1000, 10, x1, x2, y1, y2, w)
            
            # Plot classification boundary and points:
            plot(X, y, x1, x2, y1, y2)

    return (E_ins, E_outs)

def perceptronLearningAlgorithm(numSimulations, N, x1, x2, y1, y2, w = None, maxIter = 10):
    print('Started running PLA...')
    iters = []
    for _ in xrange(numSimulations):
        # Generate data-set:
        X = (b - a) * np.random.rand(N, d) + a
        y = np.array([label(X[j], x1, x2, y1, y2) for j in xrange(N)])

        hasMisclassifiedSample = True
        iter = 0
        while (hasMisclassifiedSample and (iter < maxIter)):
            hasMisclassifiedSample = False
            for i in xrange(X.shape[0]):
                if (y[i] != sign(np.dot(X[i], w))):
                    hasMisclassifiedSample = True
                    w = w + y[i] * X[i]
            iter += 1
        iters += [iter]
    avg_iter = np.average(iters)
    print('PLA converged in %0.5f iterations (for %i runs)' % (avg_iter, numSimulations))
    print('PLA convergence overview: %s' % iters)
    return iters

print('Started linear regression simulation...')
E_ins, E_outs = runSimulations(numSimulations, N, N_test)
E_in_avg = np.average(E_ins)
E_out_avg = np.average(E_outs)
print('In-sample error:     %0.5f' % E_in_avg)
print('Out-of-sample error: %0.5f' % E_out_avg)

