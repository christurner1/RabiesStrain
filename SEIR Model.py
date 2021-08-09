# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 13:29:17 2021

@author: chris
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

#import pandas as pd

"""
# reading csv files
Eyam_Data =  pd.read_csv('eyam.data', sep=",")
#print(Eyam_Data)

# initial number of Susceptibles
S_0 = Eyam_Data.iloc[0,1] 
#print(S_0)

# initial number of infectives
I_0 = Eyam_Data.iloc[0,2]
#print(I_0)
"""

#%%
"""
This is my test SEIR model for the Eyam data. Need to find appropriate beta, gamma, mu values and proper scale 
for the plot.
"""

# Total population, N.
N = 1000
# Initial number of infected, exposed, and recovered individuals, I0, E0, and R0.
I0, E0, R0 = 1, 0, 0
# Everyone else, S0, is susceptible to infection initially.
S0 = N - I0 - R0 - E0
# Contact rate, beta, mean recovery rate, gamma, death rate, mu, and latency (in 1/days).
beta, gamma, mu, a = 0.2, 1./10, .3/10, .4/10
# A grid of time points (in days)
t = np.linspace(0, 160, 160)

# The SEIR model differential equations.
def deriv(y, t, N, beta, gamma, mu, a):
    S, I, R, E = y
    dSdt = mu*N - mu*S - beta * S * I / N
    dEdt = beta * S * I / N - (mu + a) * E
    dIdt = a * E - (gamma + mu) * I
    dRdt = gamma * I - mu * R
    return dSdt, dEdt, dIdt, dRdt

# Initial conditions vector
y0 = S0, E0, I0, R0
# Integrate the SIR equations over the time grid, t.
ret = odeint(deriv, y0, t, args=(N, beta, gamma, mu, a))
S, E, I, R = ret.T

"""
Need to work on this plot... For now, commenting out. 
"""


# Plot the data on four separate curves for S(t), E(t), I(t) and R(t)
fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
ax.plot(t, S/N, 'b', alpha=0.5, lw=2, label='Susceptible')
ax.plot(t, E/N, 'k', alpha=0.5, lw=2, label='Exposed')
ax.plot(t, I/N, 'r', alpha=0.5, lw=2, label='Infected')
ax.plot(t, R/N, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
ax.set_xlabel('Time /days')
ax.set_ylabel('Number')
ax.set_ylim(0,1.2)
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
plt.title('SEIR Model')
plt.show()

