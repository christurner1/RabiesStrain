# -*- coding: utf-8 -*-
"""
Created on Thu May 27 11:21:09 2021

@author: chris
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import odeint, solve_ivp
from lmfit import minimize, Parameters, report_fit

# reading csv files
Eyam_Data =  pd.read_csv('eyam.data', sep=",")
#print(Eyam_Data)

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

''' 
#This portion works and produces a great graphic along with results, but we give it all of the parameters. 
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
#This produces a plot. However, the time steps appear to be off. Need to recheck...
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

'''
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
#N = S_0 + I_0 
N = 1000


# Initial number of infected and recovered individuals, I0 and R0.
I0, R0 = 1, 0

# Everyone else, S0, is susceptible to infection initially.
S0 = N - I0 - R0

# Contact rate, beta, and mean recovery rate, gamma, (in 1/days) for each strain.
betatrue, gammatrue = 0.25, .12 # Using these as the true values to build my test data set

# A grid of time points (in days)
t = np.linspace(0, 160, 20)
t_span = (0,Time) # using this for the IVP solver
#%%   

def f(y, t, ps):
    """
    The SIR model differential equations with parameters built-in. Will adjust for larger, more complex models.
    """
    try:
        beta = ps['beta'].value
        gamma = ps['gamma'].value

    except:
        beta, gamma = ps

    S, I, R = y
        
    dSdt = -beta * S * I/N
    dIdt = beta * S * I/N  - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt


# Integrate the SIR equations over the time grid, t. A function is easier to use when trying to calculate residuals.
def g(t, y0, ps):
    """
    The solution to the ODE f(y, t, ps) from above. Shouldn't have to change this going forward' 
    """
    ''' 
    This is my attempt at using an IVP solver. I'm having issues at the moment. Something about 'cannot unpack non-iterable float object'
    '''
    #ret = solve_ivp(f, t, y0, args = (ps,), dense_output=True)
    ret = odeint(f, y0, t, args = (ps,))
    return ret

def residual(ps, ts, data):
    y0 = ps['S0'].value, ps['I0'].value, ps['R0'].value
    model = g(ts, y0, ps)
    return (model - data).ravel()

#%%
"""
This is simply build test data to verify that the residual function is working correctly.
"""


x0 = np.array([S0, I0, R0])
true_params = np.array((betatrue, gammatrue))
data = g(t, x0, true_params)
data += np.random.normal(size=data.shape,scale = 45)

S, I, R = data.T
#%%


#print(data)
#print(data.size)


#%% 
# Setting parameters. Note that we can place bounds on values if need be.
params = Parameters()
params.add('beta', value = 0.01, min = 0, max = 1) # contact rate
params.add('gamma', value = 0.01, min = 0, max = 1) # mean recovery rate
params.add('S0', value = S0) # Initial number of susceptibles
params.add('I0', value = I0) # Initial number of infectives
params.add('R0', value =R0) # Initial number of recovered, typically 0


#%%
"""
There appears to be an issue with this particular solver. It is outdated. Still showing an excess work warning. I'm attempting to use another stiff 
ODE solver vode. Still ironing out the wrinkles. Maybe an IVP solver?
"""

# x0 = [S0, I0, R0]
# print(type(g(t,x0, params)))
# print(type(data))


# fit model and find predicted values
result = minimize(residual, params, args=(t, data), method='Nelder-Mead')
final = data + result.residual.reshape(data.shape)


# Plot the data on three separate curves for S(t), I(t) and R(t)
fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
#ax.plot(t, S, 'b', alpha=0.5, lw=2, label='Susceptible')
#ax.plot(t, I, 'r', alpha=0.5, lw=2, label='Infected')
#ax.plot(t, R, 'g', alpha=0.5, lw=2, label='Recovered with immunity')

ax.plot(t, data, 'o')
ax.plot(t, final, '-', color = 'k', linewidth=2)

ax.set_xlabel('Time')
ax.set_ylabel('Number')
ax.set_ylim(0,1100)
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.grid(b=True, which='major', c='w', lw=2, ls='-')
#legend = ax.legend()
#legend.get_frame().set_alpha(0.5)
#for spine in ('top', 'right', 'bottom', 'left'):
#    ax.spines[spine].set_visible(False)
plt.title("SIR Model (Parameter Fitting)")
plt.show()



#display fitted statistics
report_fit(result)

