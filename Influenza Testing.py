# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 16:29:05 2021

@author: chris
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import integrate, optimize
from lmfit import minimize, Parameters, report_fit

#%%

InfData = np.array([3,8,28,75,221,291,255,235,190,126,70,28,12,5])

# Using this to make each year in the data
Time = np.linspace(1, 14, 14) #this would be my x data

#%% Preliminaries prior to model validation.
"""
There is probably a faster way to evaluate each province, but for now this is the way I am doing it.
"""

# Using the average population total for our N
N = 763

# Initial number of infected and recovered individuals, I0 and R0.
I0, R0 = InfData[0], 0

# Initial number of Suscepitble , S0
S0 = N - I0 - R0

y0 = [S0, I0, R0] #just placing the intial values in matrix form 
# Contact rate, beta, and mean recovery rate, gamma.
beta, gamma = 91.25/365 , 10/356# average infection rate is 1-3 months in humans, death occurs 2-10 days after symptoms

#%% 
# Setting parameters. Note that we can place bounds on values if need be.
params = Parameters()
params.add('beta', value = beta, min = 0, max = 10000) # contact rate
params.add('gamma', value = gamma, min = 0, max = 10000) # mean recovery rate
params.add('S0', value = S0, vary = False) # Initial number of susceptibles
params.add('I0', value = I0, vary = False) # Initial number of infectives
params.add('R0', value =R0, vary = False) # Initial number of recovered, typically 0

#%%
def SIR(y, t, paras):
    """
    Here is the SIR model.
    """

    S = y[0]
    I = y[1]
        
    try:
        beta = paras['beta'].value
        gamma = paras['gamma'].value

    except KeyError:
        beta, gamma = paras
    # the model equations
    dSdt = -beta * S * I/N
    dIdt = beta * S * I/N  - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

def g(t, x0, paras):
   """
   Solution to the SIR ODE with initial condition S(0) = S0, I(0) = I0
   """
   ret = integrate.odeint(SIR, x0, t, args = (paras, ))
   return ret

def residual(ps, ts, infdata):
    y0 = ps['S0'].value, ps['I0'].value, ps['R0'].value
    model = g(ts, y0, ps)
    return (model[:,1] - infdata).ravel(),# (model[:, 0] - popdata).ravel()

#%%
# fit model
results_infected = minimize(residual, params, args = (Time, InfData), method = 'Nelder-Mead') 

# check results of the fit
data_fitted = InfData + results_infected.residual.reshape(InfData.shape)

# Plot the data on three separate curves for S(t), I(t) and R(t)
fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
#ax.plot(t, S, 'b', alpha=0.5, lw=2, label='Susceptible')
#ax.plot(t, I, 'r', alpha=0.5, lw=2, label='Infected')
#ax.plot(t, R, 'g', alpha=0.5, lw=2, label='Recovered with immunity')

#ax.plot(Time, GuangxiPopData * 10000, 'o')
ax.plot(Time, InfData, 'o')
ax.plot(Time, data_fitted, '-', color = 'k', linewidth=2)
#ax.plot(Time, pop_fitted, '-', color = 'k', linewidth=2) 

ax.set_xlabel('Days')
ax.set_ylabel('Number')
# ax.set_ylim(0,150000000)
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.grid(b=True, which='major', c='w', lw=2, ls='-')
#legend = ax.legend()
#legend.get_frame().set_alpha(0.5)
#for spine in ('top', 'right', 'bottom', 'left'):
#    ax.spines[spine].set_visible(False)
plt.title("Boarding School Flu (1978)")
plt.show()

#print(GuangdongPopData.shape)

#display fitted statistics
report_fit(results_infected)
