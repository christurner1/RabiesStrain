# -*- coding: utf-8 -*-
"""
Created on Thu May 27 11:21:09 2021

@author: chris
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import odeint
from lmfit import minimize, Parameters, report_fit

# reading csv files
Eyam_Data =  pd.read_csv('eyam.data', sep=",")
print(Eyam_Data)

#labeling the data columns
Eyam_Data.columns = ['Days', 'Susceptibles', 'Infectives']

# Seperating the data into individual columns for comparison for the residuals.
Sus = Eyam_Data['Susceptibles']
Inf = Eyam_Data['Infectives']

# initial number of Susceptibles
S_0 = Eyam_Data.iloc[0,1] 
#print(S_0)

# final number of Susceptibles
S_inf = Eyam_Data.iloc[7,1]
#print(S_inf)

# initial number of infectives
I_0 = Eyam_Data.iloc[0,2]
#print(I_0)

# Final recovered value, needed for a gamma approx.
R_inf = S_0 - Eyam_Data.iloc[0,2]
#print(R_inf)

# total time. The plus one is so that our initial time is at 0
Time = Eyam_Data.iloc[7,0] - Eyam_Data.iloc[0,0] + 1 
#print(Time)

#%% Preliminaries prior to model validation.
"""
This is my test SIR model for the Eyam data. Beta and gamma values are given in the textbook.  
"""

# Total population, N.
N = S_0 + I_0 

# Initial number of infected and recovered individuals, I0 and R0.
I0, R0 = I_0, 0
# Everyone else, S0, is susceptible to infection initially.
S0 = N - I0 - R0
# Contact rate, beta, and mean recovery rate, gamma.
beta, gamma = 0.0178, 2.73 #given in textbook

 

#%% The actual model itself
'''
This produces a plot. However, the time steps appear to be off. Need to recheck...
'''

# A grid of time points (in days)
t = np.linspace(0, Time, Time)

# The SIR model differential equations, as shown in textbook.
def derivtest(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I 
    dIdt = beta * S * I  - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

# Initial conditions vector
y0 = S0, I0, R0
# Integrate the SIR equations over the time grid, t.
ret = odeint(derivtest, y0, t, args=(N, beta, gamma))
S, I, R = ret.T

# Plot the data on three separate curves for S(t), I(t) and R(t)
fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
ax.plot(t, S, 'b', alpha=0.5, lw=2, label='Susceptible')
ax.plot(t, I, 'r', alpha=0.5, lw=2, label='Infected')
ax.plot(t, R, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
ax.set_xlabel('Time (days)')
ax.set_ylabel('Number')
ax.set_ylim(0,300)
ax.set_xlim(0,5)
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
plt.title('Eyam Data')
plt.show()


#%% Preliminaries prior to model validation. Attempting to curve fit using lmfit
"""
This is my test SIR model for the Eyam data. Using a built-in curve fitting function minimize and the data itself. Actively working on this
"""

# The susceptible data from the Eyam Data
Sus = Eyam_Data['Susceptibles'] 
#print(Sus)

# The infective data from the Eyam Data
Inf = Eyam_Data['Infectives']
#print(Inf)

Days = Eyam_Data['Days']
#print(Days)

# Total population, N.
N = S_0 + I_0 

# Initial number of infected and recovered individuals, I0 and R0.
I0, R0 = I_0, 0

# Everyone else, S0, is susceptible to infection initially.
S0 = N - I0 - R0

"""
Still need a proper method for an initial guess for gamma and beta. I have a ratio, but no clear way to estimate gamma.
"""

'''
# Contact rate, beta, and mean recovery rate, gamma.
beta, gamma = 0.05, 3 # These are initial guesses
'''

'''
# Initial conditions vector
y0 = S0, I0, R0
initial_conditions = [S0, I0, R0]
'''

    
# Setting parameters. Note that we can place bounds on values if need be.
params = Parameters()
params.add('beta', value = 0.05, min = 0, max = 10) # contact rate
params.add('gamma', value = 3, min = 0, max = 10) # mean recovery rate
params.add('S0', value = S_0) # Initial number of susceptibles
params.add('I0', value = I_0) # Initial number of infectives
params.add('R0', value = 0) # Initial number of recovered, typically 0


# A grid of time points (in days)
t = np.linspace(0, Time, Time)

def f(y, t, ps):
    """
    The SIR model differential equations with parameters built-in. Will adjust for larger, more complex models.
    """
    beta = ps['beta'].value
    gamma = ps['gamma'].value
    dSdt = -beta * S * I 
    dIdt = beta * S * I  - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt


# Integrate the SIR equations over the time grid, t. A function is easier to use when trying to calculate residuals.
def g(y, t, ps):
    """
    The solution to the ODE f(y, t, ps) from above. Shouldn't have to change this going forward' 
    """
    ret = odeint(f, y, t, args = (ps,))
    return ret

"""
Probably need to build a for loop to compare the sparse Eyam Data with the results from the ODE
"""
print(ret)
print(ret.shape)
#S, I, R = ret.T

'''
# To calculate the residuals of the data versus what the model predicts. Need this to minimize error and for a better fit. 
# This compares the modeled infectives versus actual infectives. 
def residuals(ps, t, data):
    y = ps['S0'].value, ps['I0'].value, ps['R0'].value
    sol = g(y, t, params)        
    return sol #(sol-data).ravel() # need to either fix the data to match the same size as sol or some other method
'''

#residuals(params, t, Sus)
#result = minimize(residuals, params, args = (t, Eyam_Data), method = 'leastsq') #need to fix the Eyam_Data so that a comparison can happen
#report_fit(result)
    
#print(residuals(params, initial_conditions, t, Eyam_Data))


#%% The model that works and should not be changed in any way.
"""
This is a good, working SIR model with given parameters. This model does not use vital dynamics, i.e. no birth or death rate for a population. Don't Change!!!
"""

# Total population, N.
N = 1000
# Initial number of infected and recovered individuals, I0 and R0.
I0, R0 = 1, 0
# Everyone else, S0, is susceptible to infection initially.
S0 = N - I0 - R0
# Contact rate, beta, and mean recovery rate, gamma, (in 1/days).
beta, gamma = 0.2, 1./10 
# A grid of time points (in days)
t = np.linspace(0, 160, 160)

# The SIR model differential equations.
def deriv(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

# Initial conditions vector
y0 = S0, I0, R0
# Integrate the SIR equations over the time grid, t.
ret = odeint(deriv, y0, t, args=(N, beta, gamma))
S, I, R = ret.T

# Plot the data on three separate curves for S(t), I(t) and R(t)
fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
ax.plot(t, S/1000, 'b', alpha=0.5, lw=2, label='Susceptible')
ax.plot(t, I/1000, 'r', alpha=0.5, lw=2, label='Infected')
ax.plot(t, R/1000, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
ax.set_xlabel('Time /days')
ax.set_ylabel('Number (1000s)')
ax.set_ylim(0,1.2)
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
plt.title("SIR Model")
plt.show()