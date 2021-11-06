# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 16:25:39 2021

@author: chris
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


#%% The model that works and should not be changed in any way.
"""
This is a good, working SIR model with given parameters. This model does not use vital dynamics, i.e. no birth or death rate for a population. Don't Change!!!
"""

# Total population, N.
N = 100
# Initial number of infected and recovered individuals, I0 and R0.
I_10, I_20, R0 = 1, 1, 0
# Everyone else, S0, is susceptible to infection initially.
S0 = N - I_10 -I_20 - R0
# Contact rate, beta, and mean recovery rate, gamma, (in 1/days).
beta1, beta2, gamma1, gamma2 = .8, .6, 1./5, .3 
# A grid of time points (in days)
t = np.linspace(0, 30, 30)

# The SIR model differential equations.
def deriv(y, t, N, beta1, beta2, gamma1, gamma2):
    S, I_1, I_2, R = y
    dSdt = -beta1 * I_1 * S/N  + -beta2 * I_2 * S/N  # I changed this to a plus instead of a minus
    dI_1dt = beta1 * I_1 * S/N - gamma1 * I_1 
    dI_2dt = beta2 * I_2 * S/N - gamma2 * I_2
    dRdt = gamma1 * I_1 + gamma2 * I_2
    return dSdt, dI_1dt, dI_2dt, dRdt

#%%
# Initial conditions vector
y0 = S0, I_10, I_20, R0
# Integrate the SIR equations over the time grid, t.
ret = odeint(deriv, y0, t, args=(N, beta1, beta2, gamma1, gamma2))
S, I_1, I_2, R = ret.T

# Plot the data on three separate curves for S(t), I(t) and R(t)
fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
ax.plot(t, S, 'b', alpha=0.5, lw=2, label='Susceptible')
ax.plot(t, I_1, 'r', alpha=0.5, lw=2, label='Infected (Strain 1)')
ax.plot(t, I_2, 'k', alpha=0.5, lw=2, label='Infected (Strain 2)')
ax.plot(t, R, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
ax.set_xlabel('Time')
ax.set_ylabel('Number')
ax.set_ylim(0,110)
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
plt.title("The Basic SIR Model")
plt.show()