# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 17:37:05 2021

@author: chris

The purpose of this code is to apply a dog-hybrid SIR model to the provincial rabies
data and produce estimates of the model parameteres over the time span of
data available.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import integrate
from lmfit import minimize, Parameters, report_fit

def define_provinces():
    ''' This function just creates a list as global variable with all the
        province names. '''

    global provinces


    provinces = ['Guangxi', 'Hunan', 'Guizhou', 'Guangdong', 'Jiangxi',
                 'Hubei', 'Jiangsu', 'Henan', 'Sichuan', 'Anhui', 'Shandong',
                 'Hebei', 'Chongqing', 'Yunnan', 'Zhejiang', 'Hainan',
                 'Fujian', 'Shanxi', 'Shaanxi', 'Tianjin', 'Inner Mongolia',
                 'Shanghai', 'Beijng', 'Liaoning', 'Jilin', 'Heilongjiang',
                 'Xinjiang', 'Gansu', 'Tibet', 'Ningxia', 'Qinghai']

def load_data():
    '''reading csv files. The excel read gave me some weird outputs.
       Just converted the .xlsx file to a .csv file. Won't work
       with multiple sheet excel workbooks '''

    global N, NH, ND, GuangdongData, params, ps, Time
    global province_name, province_data

    #ChinaDataTest = pd.read_excel('1996-2020 rabies province.xlsx')
    china_rabies_data =  pd.read_csv('1996-2020 rabies province.csv')
    china_pop_data = pd.read_excel('Chinese_population_data.xlsx')

    # fixing the data to be better readable

    #take the data, now minus the header
    china_rabies_data = china_rabies_data[1:]

    # removing the rows with NaN values.
    china_rabies_data = china_rabies_data.dropna(how = "all", axis = 0)
    china_pop_data = china_pop_data.dropna(how = 'all', axis = 0)

    # removing the columns with NaN values.
    china_rabies_data = china_rabies_data.dropna(how = 'all', axis = 'columns')
    china_pop_data = china_pop_data.dropna(how = 'all', axis = 'columns')

    #print(ChinaPopTest)
    #Renaming the column names
    china_rabies_data.columns=["Province",'1996','1997','1998', '1999', '2000',
                               '2001', '2002', '2003','2004', '2005','2006',
                               '2007', '2008', '2009', '2010', '2011', '2012',
                               '2013', '2014', '2015' ,'2016', '2017', '2018',
                               '2019', '2020']

    china_pop_data.columns = ["Province",'1996','1997','1998', '1999', '2000',
                              '2001', '2002', '2003','2004', '2005','2006',
                              '2007', '2008', '2009', '2010', '2011', '2012',
                              '2013', '2014', '2015' ,'2016', '2017', '2018',
                              '2019']

    # Renaming for convenience
    cd = china_rabies_data
    cp = china_pop_data

    # removing the last column so that ChinaDataTest and ChinaPopTest match.
    cd.drop(columns = cd.columns[-1], axis = 1, inplace = True)

    # populate infections per province
    province_infected = dict()
    for p_name in provinces:
        province_infected[p_name] = (cd.loc[cd['Province'] == p_name]).to_numpy()

    # populate province data
    province_data = dict()
    for p_name in provinces:
        province_data[p_name] = np.delete(province_infected[p_name],[0])

    # populate province populations
    province_pop = dict()
    for p_name in provinces:
        province_pop[p_name] = (cp.loc[cp['Province'] == p_name]).to_numpy()

    # populate province population data
    province_pop_data = dict()
    
    global bad_provinces
    bad_provinces = list()
    
    for p_name in provinces:
        if len(province_pop[p_name]) > 0:
            province_pop_data[p_name] = np.delete(province_pop[p_name],[0])
        else:
            print("\tThere does not appear to be population data for " + p_name)
            bad_provinces.append(p_name)
            
    print(" ") # just a line break for prettiness

    #
    # Using this to make each year in the data
    Time = np.linspace(1996, 2019, 24) #this would be my x data

    # Preliminaries prior to model validation.

    # Using the average population total for our N
    NH = np.average(province_pop_data[province_name] * 10000)

    # Initial number of infected and recovered individuals, I0 and R0.
    I0, R0 = province_data[province_name][0], province_data[province_name][0]
    ID0, RD0 = 394000, 394000 # Number of infected dogs
    
    # Initial number of Suscepitble , S0
    S0 = NH - I0 - R0
    SD0 = 3940000 # Number of susceptible dogs
    
    ND = SD0 + ID0 + RD0 # total dog population
    
    # The total dog and human population in Yunnan
    N = NH+ND

    #y0 = [S0, I0, R0] #just placing the intial values in matrix form 

    # Birth rate and natural death rate for dogs
    birth_dogs = 12
    death_dogs = 6

    # Birth rate and natural death rate for humans
    birth = .5
    death= .25

    # Contact rate, beta, and mean recovery rate, gamma.
    beta, gamma = 15, 15 # average infection rate is 1-3 months in humans, death occurs 2-10 days after symptoms
    beta_dogs, gamma_dogs = 75, 75 #infection rate for dog-dog interaction, "recovery rate" for dog-dog interaction
    

    # Setting parameters. Note that we can place bounds on values if need be.
    params = Parameters()
    params.add('birth_dogs', value = birth_dogs, min = 0) # birth rate for dogs per year
    params.add('death_dogs', value = death_dogs, min = 0) # natural death rate for dogs per year
    params.add('birth', value = birth, min = 0) # birth rate for humans per year
    params.add('death', value = death, min = 0) # natural death rate for humans per year
    params.add('beta', value = beta, min = 0) # contact rate for dog-human interaction
    params.add('gamma', value = gamma, min = 0) # mean recovery rate for dog-human interaction
    params.add('beta_dogs', value = beta_dogs, min = 0) # contact rate for dog-dog interaction
    params.add('gamma_dogs', value = gamma_dogs, min = 0) # mean recovery rate for dog-dog interaction
    params.add('S0', value = S0, vary = False) # Initial number of susceptibles
    params.add('I0', value = I0, vary = False) # Initial number of infectives
    params.add('R0', value =R0, vary = False) # Initial number of recovered, typically 0
    params.add('SD0', value = SD0, min = 0, max = 10000000) # Initial number of susceptible dogs
    params.add('ID0', value = ID0, min = 0, max = 1000000) # Initial number of infected dogs
    params.add('RD0', value =RD0, min = 0, max = 1000000) # Initial number of "recovered" dogs, typically 0

def SIR(y, t, paras):
    """
    Here is the SIR model.
    """

    S = y[0]
    I = y[1]
    SD = y[3]
    ID = y[4]
        
    try:
        beta = paras['beta'].value
        gamma = paras['gamma'].value
        beta_dogs = paras['beta_dogs'].value
        gamma_dogs = paras['gamma_dogs'].value
        birth_dogs = paras['birth_dogs'].value
        death_dogs = paras['death_dogs'].value
        birth = paras['birth'].value
        death = paras['death'].value
    except KeyError:
        beta, gamma, beta_dogs, gamma_dogs, birth_dogs, death_dogs, birth, death = paras
        
    # the model equations
    dSdt = birth * NH - death * S - beta * S * ID/N 
    dIdt = beta * S * ID/N  - gamma * I 
    dRdt = gamma * I
    dSDdt = birth_dogs * ND - death_dogs * SD - beta_dogs * ID *SD /ND
    dIDdt = beta_dogs * ID *SD/ND - gamma_dogs * ID
    dRDdt = gamma_dogs * ID
    return [dSdt, dIdt, dRdt, dSDdt, dIDdt, dRDdt]

def g(t, x0, paras):
   """
   Solution to the SIR ODE with initial condition S(0) = S0, I(0) = I0
   """
   ret = integrate.odeint(SIR, x0, t, args = (paras, ))
   return ret

def residual(ps, ts, infdata):
    y0 = ps['S0'].value, ps['I0'].value, ps['R0'].value, ps['SD0'].value, ps['ID0'].value, ps['RD0'].value
    model = g(ts, y0, ps)
    return (model[:,1] - infdata).ravel(),# (model[:, 0] - popdata).ravel()


def main():
    ''' This is the main function for testing and running this code. '''

    load_data()

    # fit model
    results_infected = minimize(residual, params, args = (Time, province_data[province_name]),
                                method = 'Nelder-Mead')

    # check results of the fit
    data_fitted = province_data[province_name] + results_infected.residual.reshape(province_data[province_name].shape)

    # Plot the data on three separate curves for S(t), I(t) and R(t)
    fig = plt.figure(facecolor='w')
    ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)

    #ax.plot(Time, GuangdongPopData * 10000, 'o')
    ax.plot(Time, province_data[province_name], 'o')
    ax.plot(Time, data_fitted, '-', color = 'k', linewidth=2)

    ax.set_xlabel('Year')
    ax.set_ylabel('Number')
    #ax.set_ylim(0,15000)
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    ax.grid(visible=True, which='major', c='w', lw=2, ls='-')

    plt.title(province_name)
    plt.show()

    #display fitted statistics
    report_fit(results_infected)

if __name__ == '__main__':
    
    define_provinces()

    global province_name
    
    # Method to pick your province
    i = 0
    for p_name in provinces:
        print(str(i) + "  " + p_name)
        i += 1
    
    i_province = input("Type the number of province you'd like to see displayed and hit enter: ")
    province_name = provinces[int(i_province)]
    print("You chose " + province_name)
    
    main()