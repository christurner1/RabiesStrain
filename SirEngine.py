#!/usr/bin/python

from pylab import *
import csv
import sys

def stepForwardOneTimeStep(alpha, beta, sus0, inf0, rec0):
    deltaS = -beta * sus0 * inf0
    deltaR = alpha * inf0
    deltaI = -deltaS - deltaR

    sus = sus0 + deltaS
    inf = inf0 + deltaI
    rec = rec0 + deltaR

    returnPoint = [sus, inf, rec] 
    return returnPoint

def runFullSim(alpha, beta, sus0, inf0, rec0, firstDay, lastDay):
    numberDivsPerDay  = 100.0 
    numberOfDays      = lastDay - firstDay + 1
   
    dayVector         = linspace(firstDay, lastDay, numberOfDays)
    susceptibleArray  = [sus0]
    infectedArray     = [inf0]
    recoveredArray    = [rec0]

    sus = sus0
    inf = inf0
    rec = rec0

    for timeValue in range(numberOfDays - 1):
        for j in range(int(numberDivsPerDay)):
            nextPop = stepForwardOneTimeStep(alpha/numberDivsPerDay,
                                             beta /numberDivsPerDay,
                                             sus,
                                             inf,
                                             rec)

            sus = nextPop[0]
            inf = nextPop[1]
            rec = nextPop[2]

        susceptibleArray += [sus]
        infectedArray    += [inf]
        recoveredArray   += [rec]

    return [dayVector, susceptibleArray, infectedArray, recoveredArray]
