# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 10:36:40 2021

@author: chris
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from lmfit import  Parameters, minimize, report_fit

#%%
'''
This block will be where we download the data and do manipulations prior to the paramater fitting.
'''

# Total population, N.
N = 1000
# Initial number of infected and recovered individuals, I_10, I_20 and R0.
I_10, I_20, R0 = 1, 1, 0
# Everyone else, S0, is susceptible to infection initially.
S0 = N - I_10 - I_20 - R0
# Contact rate, beta, and mean recovery rate, gamma, (in 1/days) for each strain.
beta1true, gamma1true, beta2true, gamma2true = 0.2, 1./10 , 0.25, 1./10  # Using these as the true values to build my test data set
# A grid of time points (in days)
t = np.linspace(0, 160, 160)


#%%
# The Two-Strain SIR model differential equations. 
def f(y, t, ps):
    """
    The Two-Strain SIR model differential equations with parameters built-in. Will adjust for larger, more complex models.
    """     
    
    try:
        beta1 = ps['beta1'].value
        beta2 = ps['beta2'].value
        gamma1 = ps['gamma1'].value
        gamma2 = ps['gamma2'].value
    except:
        beta1, beta2, gamma1, gamma2 = ps

    S, I_1, I_2, R = y
    dSdt = -beta1 * I_1 * S/N  + -beta2 * I_2 * S/N  # I changed this to a plus instead of a minus
    dI_1dt = beta1 * I_1 * S/N - gamma1 * I_1 
    dI_2dt = beta2 * I_2 * S/N - gamma2 * I_2
    dRdt = gamma1 * I_1 + gamma2 * I_2
    return [dSdt, dI_1dt, dI_2dt, dRdt]


# Integrate the SIR equations over the time grid, t. A function is easier to use when trying to calculate residuals.
def g(t, y0, ps):
    """
    The solution to the ODE f(y, t, ps) from above. Shouldn't have to change this going forward' 
    """
    ret = odeint(f, y0, t, args = (ps,))
    return ret

# Comparing the results from the ode solver to the data given.
def residual(ps, ts, data):
    y0 = ps['S0'].value, ps['I10'].value, ps['I20'].value, ps['R0'].value
    model = g(ts, y0, ps)
    return (model - data).ravel()

#%%
"""
This is simply build test data to verify that the residual function is working correctly.
"""

x0 = np.array([S0, I_10, I_20, R0])
true_params = np.array((beta1true, beta2true, gamma1true, gamma2true))
data = g(t, x0, true_params)
data += np.random.normal(size=data.shape)

S, I_1,I_2, R = data.T
#%%

# Setting parameters. Note that we can place bounds on values if need be.
params = Parameters()
params.add('beta1', value = 0.05, min = 0, max = 1) # contact rate for strain 1
params.add('beta2', value = 0.05, min = 0, max = 1) # contact rate for strain 2
params.add('gamma1', value = .05, min = 0, max = 2) # mean recovery rate for strain 1
params.add('gamma2', value = .5, min = 0, max = 2) # mean recovery rate for strain 2
params.add('S0', value = S0) # Initial number of susceptibles
params.add('I10', value = I_10) # Initial number of infectives for strain 1
params.add('I20', value = I_20) # Initial number of infectives for strain 1
params.add('R0', value = R0) # Initial number of recovered, typically 0
#%%
"""
There appears to be an issue with the particular ode solver lsoda which leads to a warning about excess work. I'm attempting to use another non-stiff 
ODE solver vode. Still ironing out the wrinkles. Maybe an IVP solver?
"""
# fit model and find predicted values
result = minimize(residual, params, args=(t, data), method='leastsq')
final = data + result.residual.reshape(data.shape)


# display fitted statistics
report_fit(result)
#%%%
# Plot the data on three separate curves for S(t), I(t) and R(t)
fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
#ax.plot(t, S, 'b', alpha=0.5, lw=2, label='Susceptible')
#ax.plot(t, I_1, 'r', alpha=0.5, lw=2, label='Infected (Strain 1)')
#ax.plot(t, I_2, 'k', alpha=0.5, lw=2, label='Infected (Strain 2)')
#ax.plot(t, R, 'g', alpha=0.5, lw=2, label='Recovered with immunity')

plt.plot(t, data, 'o')
plt.plot(t, final, '-', color = 'k' , linewidth=2);


ax.set_xlabel('Time')
ax.set_ylabel('Number')
ax.set_ylim(0,1100)
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
plt.title("Two-Strain SIR Model")
plt.show()



