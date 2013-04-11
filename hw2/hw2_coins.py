from __future__ import division
import numpy as np
import sys

print(sys.version + '\n')

# By convention:
# head = 1
# tail = 0

def runSimulation(numSimulations = 100000, numCoins = 1000, numFlips = 10):
    nuFirst = []
    nuRand = []
    nuMin = []
    
    for _ in xrange(numSimulations):
        coins = np.random.randint(2, size = (numCoins,numFlips))
        numHeads = coins.sum(axis = 1)
        
        indexMin = np.unravel_index(numHeads.argmin(), numHeads.shape)
        indexRand = np.random.randint(numCoins)

        nuFirst += [np.average(coins[0])]
        nuRand += [np.average(coins[indexRand])]
        nuMin += [np.average(coins[indexMin])]
    
    return (nuFirst, nuRand, nuMin)

print('Started coins simulation...')
nuFirst, nuRand, nuMin = runSimulation()
print('Average first coin: %0.5f' % np.average(nuFirst))
print('Average rand coin:  %0.5f' % np.average(nuRand))
print('Average min coin:   %0.5f' % np.average(nuMin))
