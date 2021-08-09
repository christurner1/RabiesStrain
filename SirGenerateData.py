#!/usr/bin/python

import math
import scipy
from scipy import integrate
from numpy import *
from pylab import *
from SirEngine import runFullSim

def phaseMetric(alpha, beta, s, i):
    c = s + i - alpha * math.log(s) / beta
    return c

# Initialize parameters for simulation
alpha = 0.10  
beta  = 0.0007
sus0  = 500.0
inf0  = 10.0
rec0  = 0
numPoints = 100

con0 = phaseMetric(alpha, beta, sus0,inf0)

sus = sus0
inf = inf0
rec = rec0

simulatedPopulation = runFullSim(alpha, beta, sus0,
                                 inf0,rec0,1, numPoints)

time         = simulatedPopulation[0]
susceptible  = simulatedPopulation[1]
infected     = simulatedPopulation[2]
recovered    = simulatedPopulation[3]

for timeIndex in range(len(time)):
    sus = susceptible[timeIndex]    
    inf = infected[timeIndex]
    con = phaseMetric(alpha, beta, sus, inf)
    
    print (str(timeIndex + 1)          + ", " + \
          str(susceptible[timeIndex]) + ", " + \
          str(infected[timeIndex])    + ", " + \
          str(recovered[timeIndex]))

