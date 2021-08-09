#!/usr/bin/python

import math
import scipy
from scipy import integrate
from numpy import *
from pylab import *
import csv
import sys

filename = ""

if (len(sys.argv) != 2):
   print ("usage: ./plotData.py datafilename")
   sys.exit()
else:
   filename = sys.argv[1]

data        = []
day         = []
susceptible = []
infected    = []
recovered   = []

for row in csv.reader(open(filename)):
     data.append(row)
     print(row)

for dsi in data:
     day         += [float(dsi[0])]
     susceptible += [float(dsi[1])]
     infected    += [float(dsi[2])]
     recovered   += [float(dsi[3])]

title("Input Data")

plot(day, susceptible)
plot(day, infected)
plot(day, recovered)

l = ['Susceptible','Infected','Recovered']
legend(l)

show()
