# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 17:37:05 2021

@author: chris

The purpose of this code is to apply a SIR model to the provincial rabies
data and produce estimates of the model parameteres over the time span of
data available. -- my (Jim's) text is lame here and needs some help...
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import integrate
from lmfit import minimize, Parameters, report_fit

#%%
# reading csv files. The excel read gave me some weird outputs. Just converted
# the .xlsx file to a .csv file. Won't work
# with multiple sheet excel workbooks

#ChinaDataTest = pd.read_excel('1996-2020 rabies province.xlsx')
china_rabies_data =  pd.read_csv('1996-2020 rabies province.csv')
china_pop_data = pd.read_excel('Chinese_population_data.xlsx')

#print(ChinaPop)

# fixing the data to be better readable

#takes the first row, which is the header
header =  china_rabies_data.iloc[0]
#print(header)

#take the data, now minus the header
china_rabies_data = china_rabies_data[1:]
#print(ChinaDataTest)

# removing the rows with NaN values.
china_rabies_data = china_rabies_data.dropna(how = "all", axis = 0)
china_pop_data = china_pop_data.dropna(how = 'all', axis = 0)


# removing the columns with NaN values.
china_rabies_data = china_rabies_data.dropna(how = 'all', axis = 'columns')
china_pop_data = china_pop_data.dropna(how = 'all', axis = 'columns')

#print(ChinaPopTest)
#Renaming the column names
china_rabies_data.columns=["Province",'1996','1997','1998', '1999', '2000', '2001', '2002', '2003',
                        '2004', '2005','2006', '2007', '2008', '2009', '2010', '2011', '2012',
                        '2013', '2014', '2015' ,'2016', '2017', '2018', '2019', '2020']

china_pop_data.columns = ["Province",'1996','1997','1998', '1999', '2000', '2001', '2002', '2003',
                        '2004', '2005','2006', '2007', '2008', '2009', '2010', '2011', '2012',
                        '2013', '2014', '2015' ,'2016', '2017', '2018', '2019']

# Renaming for convenience
cd = china_rabies_data
cp = china_pop_data
#print(cd)
# removing the last column so that ChinaDataTest and ChinaPopTest are mathchin
cd.drop(columns = cd.columns[-1], axis = 1, inplace = True)
#print(cd)

# There is probably an easier way to do this next section, but I know that
# this works and I'm not trying to improve speed at this moment
#%%
# Getting the data for individual provinces, just need to change the name after ==
GuangxiInfected = cd.loc[cd['Province'] == 'Guangxi']
HunanInfected = cd.loc[cd['Province'] == 'Hunan']
GuizhouInfected = cd.loc[cd['Province'] == 'Guizhou']
GuangdongInfected = cd.loc[cd['Province'] == 'Guangdong']
JiangxiInfected = cd.loc[cd['Province'] == 'Jiangxi']
HubeiInfected = cd.loc[cd['Province'] == 'Hubei']
JiangsuInfected = cd.loc[cd['Province'] == 'Jiangsu']
HenanInfected = cd.loc[cd['Province'] == 'Henan']
SichuanInfected = cd.loc[cd['Province'] == 'Sichuan']
AnhuiInfected = cd.loc[cd['Province'] == 'Anhui']
ShandongInfected = cd.loc[cd['Province'] == 'Shandong']
HebeiInfected = cd.loc[cd['Province'] == 'Hebei']
ChongqingInfected = cd.loc[cd['Province'] == 'Chongqing']
YunnanInfected = cd.loc[cd['Province'] == 'Yunnan']
ZhejiangInfected = cd.loc[cd['Province'] == 'Zhejiang']
HainanInfected = cd.loc[cd['Province'] == 'Hainan']
FujianInfected = cd.loc[cd['Province'] == 'Fujian']
ShanxiInfected = cd.loc[cd['Province'] == 'Shanxi']
ShaanxiInfected = cd.loc[cd['Province'] == 'Shaanxi']
TianjinInfected = cd.loc[cd['Province'] == 'Tianjin']
InnerMongoliaInfected = cd.loc[cd['Province'] == 'Inner Mongolia']
ShanghaiInfected = cd.loc[cd['Province'] == 'Shanghai']
BeijingInfected = cd.loc[cd['Province'] == 'Beijng']
LiaoningInfected = cd.loc[cd['Province'] == 'Liaoning']
JilinInfected = cd.loc[cd['Province'] == 'Jilin']
HeilongjiangInfected = cd.loc[cd['Province'] == 'Heilongjiang']
XinjiangInfected = cd.loc[cd['Province'] == 'Xinjiang']
GansuInfected = cd.loc[cd['Province'] == 'Gansu']
TibetInfected = cd.loc[cd['Province'] == 'Tibet']
NingxiaInfected = cd.loc[cd['Province'] == 'Ningxia']
QinghaiInfected = cd.loc[cd['Province'] == 'Qinghai']


# Converts the dataframe to a numpy array
GuangxiInfected = GuangxiInfected.to_numpy()
HunanInfected = HunanInfected.to_numpy()
GuizhouInfected = GuizhouInfected.to_numpy()
GuangdongInfected = GuangdongInfected.to_numpy()
JiangxiInfected = JiangxiInfected.to_numpy()
HubeiInfected = HubeiInfected.to_numpy()
JiangsuInfected = JiangsuInfected.to_numpy()
HenanInfected = HenanInfected.to_numpy()
SichuanInfected = SichuanInfected.to_numpy()
AnhuiInfected = AnhuiInfected.to_numpy()
ShandongInfected = ShandongInfected.to_numpy()
HebeiInfected = HebeiInfected.to_numpy()
ChongqingInfected = ChongqingInfected.to_numpy()
YunnanInfected = YunnanInfected.to_numpy()
ZhejiangInfected = ZhejiangInfected.to_numpy()
HainanInfected = HainanInfected.to_numpy()
FujianInfected = FujianInfected.to_numpy()
ShanxiInfected = ShanxiInfected.to_numpy()
ShaanxiInfected = ShaanxiInfected.to_numpy()
TianjinInfected = TianjinInfected.to_numpy()
InnerMongoliaInfected = InnerMongoliaInfected.to_numpy()
ShanghaiInfected = ShanghaiInfected.to_numpy()
BeijingInfected = BeijingInfected.to_numpy()
LiaoningInfected = LiaoningInfected.to_numpy()
JilinInfected = JilinInfected.to_numpy()
HeilongjiangInfected = HeilongjiangInfected.to_numpy()
XinjiangInfected = XinjiangInfected.to_numpy()
GansuInfected = GansuInfected.to_numpy()
TibetInfected = TibetInfected.to_numpy()
NingxiaInfected = NingxiaInfected.to_numpy()
QinghaiInfected = QinghaiInfected.to_numpy()


# Removing the first element, i.e. the string 'Guangxi'
GuangxiData = np.delete(GuangxiInfected,[0])
HunanData = np.delete(HunanInfected,[0])
GuizhouData = np.delete(GuizhouInfected,[0])
GuangdongData = np.delete(GuangdongInfected,[0])
JiangxiData = np.delete(JiangxiInfected, [0])
HubeiData =np.delete(HubeiInfected,[0])
JiangsuData = np.delete(JiangsuInfected,[0])
HenanData = np.delete(HenanInfected,[0])
SichuanData = np.delete(SichuanInfected,[0])
AnhuiData = np.delete(AnhuiInfected,[0])
ShandongData = np.delete(ShandongInfected,[0])
HebeiData = np.delete(HebeiInfected,[0])
ChongqingData = np.delete(ChongqingInfected,[0])
YunnanData = np.delete(YunnanInfected,[0])
ZhejiangData = np.delete(ZhejiangInfected,[0])
HainanData = np.delete(HainanInfected,[0])
FujianData = np.delete(FujianInfected,[0])
ShanxiData = np.delete(ShanxiInfected,[0])
ShaanxiData = np.delete(ShaanxiInfected,[0])
TianjinData = np.delete(TianjinInfected,[0])
InnerMongoliaData = np.delete(InnerMongoliaInfected,[0])
ShanghaiData = np.delete(ShanghaiInfected,[0])
BeijingData = np.delete(BeijingInfected,[0])
LiaoningData = np.delete(LiaoningInfected,[0])
JilinData = np.delete(JilinInfected,[0])
HeilongjiangData = np.delete(HeilongjiangInfected,[0])
XinjiangData = np.delete(XinjiangInfected,[0])
GansuData = np.delete(GansuInfected,[0])
TibetData = np.delete(TibetInfected,[0])
NingxiaData = np.delete(NingxiaInfected,[0])
QinghaiData = np.delete(QinghaiInfected,[0])


#%%
# Getting the data for individual provinces, just need to change the name after ==
GuangxiPop = cp.loc[cp['Province'] == 'Guangxi']
HunanPop = cp.loc[cp['Province'] == 'Hunan']
GuizhouPop = cp.loc[cp['Province'] == 'Guizhou']
GuangdongPop = cp.loc[cp['Province'] == 'Guangdong']
JiangxiPop = cp.loc[cp['Province'] == 'Jiangxi']
HubeiPop = cp.loc[cp['Province'] == 'Hubei']
JiangsuPop = cp.loc[cp['Province'] == 'Jiangsu']
HenanPop = cp.loc[cp['Province'] == 'Henan']
SichuanPop = cp.loc[cp['Province'] == 'Sichuan']
AnhuiPop = cp.loc[cp['Province'] == 'Anhui']
ShandongPop = cp.loc[cp['Province'] == 'Shandong']
HebeiPop = cp.loc[cp['Province'] == 'Hebei']
ChongqingPop = cp.loc[cp['Province'] == 'Chongqing']
YunnanPop = cp.loc[cp['Province'] == 'Yunnan']
ZhejiangPop = cp.loc[cp['Province'] == 'Zhejiang']
HainanPop = cp.loc[cp['Province'] == 'Hainan']
FujianPop = cp.loc[cp['Province'] == 'Fujian']
ShanxiPop = cp.loc[cp['Province'] == 'Shanxi']
ShaanxiPop = cp.loc[cp['Province'] == 'Shaanxi']
TianjinPop = cp.loc[cp['Province'] == 'Tianjin']
InnerMongoliaPop = cp.loc[cp['Province'] == 'Inner Mongolia']
ShanghaiPop = cp.loc[cp['Province'] == 'Shanghai']
BeijingPop = cp.loc[cp['Province'] == 'Beijing']
LiaoningPop = cp.loc[cp['Province'] == 'Liaoning']
JilinPop = cp.loc[cp['Province'] == 'Jilin']
HeilongjiangPop = cp.loc[cp['Province'] == 'Heilongjiang']
XinjiangPop = cp.loc[cp['Province'] == 'Xinjiang']
GansuPop = cp.loc[cp['Province'] == 'Gansu']
TibetPop = cp.loc[cp['Province'] == 'Tibet']
NingxiaPop = cp.loc[cp['Province'] == 'Ningxia']
QinghaiPop = cp.loc[cp['Province'] == 'Qinghai']


# Converts the dataframe to a numpy array
GuangxiPop = GuangxiPop.to_numpy()
HunanPop = HunanPop.to_numpy()
GuizhouPop = GuizhouPop.to_numpy()
GuangdongPop = GuangdongPop.to_numpy()
JiangxiPop = JiangxiPop.to_numpy()
HubeiPop = HubeiPop.to_numpy()
JiangsuPop = JiangsuPop.to_numpy()
HenanPop = HenanPop.to_numpy()
SichuanPop = SichuanPop.to_numpy()
AnhuiPop = AnhuiPop.to_numpy()
ShandongPop = ShandongPop.to_numpy()
HebeiPop = HebeiPop.to_numpy()
ChongqingPop = ChongqingPop.to_numpy()
YunnanPop = YunnanPop.to_numpy()
ZhejiangPop = ZhejiangPop.to_numpy()
HainanPop = HainanPop.to_numpy()
FujianPop = FujianPop.to_numpy()
ShanxiPop = ShanxiPop.to_numpy()
ShaanxiPop = ShaanxiPop.to_numpy()
TianjinPop = TianjinPop.to_numpy()
InnerMongoliaPop = InnerMongoliaPop.to_numpy()
ShanghaiPop = ShanghaiPop.to_numpy()
BeijingPop = BeijingPop.to_numpy()
LiaoningPop = LiaoningPop.to_numpy()
JilinPop = JilinPop.to_numpy()
HeilongjiangPop = HeilongjiangPop.to_numpy()
XinjiangPop = XinjiangPop.to_numpy()
GansuPop = GansuPop.to_numpy()
TibetPop = TibetPop.to_numpy()
NingxiaPop = NingxiaPop.to_numpy()
QinghaiPop = QinghaiPop.to_numpy()

#%%
# Removing the first element, i.e. the string 'Guangxi'
GuangxiPopData = np.delete(GuangxiPop,[0])
HunanPopData = np.delete(HunanPop,[0])
GuizhouPopData = np.delete(GuizhouPop,[0])
GuangdongPopData = np.delete(GuangdongPop,[0])
JiangxiPopData = np.delete(JiangxiPop,[0])
HubeiPopData =np.delete(HubeiPop,[0])
JiangsuPopData = np.delete(JiangsuPop,[0])
HenanPopData = np.delete(HenanPop,[0])
SichuanPopData = np.delete(SichuanPop,[0])
AnhuiPopData = np.delete(AnhuiPop,[0])
ShandongPopData = np.delete(ShandongPop,[0])
HebeiPopData = np.delete(HebeiPop,[0])
ChongqingPopData = np.delete(ChongqingPop,[0])
YunnanPopData = np.delete(YunnanPop,[0])
ZhejiangPopData = np.delete(ZhejiangPop,[0])
HainanPopData = np.delete(HainanPop,[0])
FujianPopData = np.delete(FujianPop,[0])
ShanxiPopData = np.delete(ShanxiPop,[0])
#ShaanxiPopData = np.delete(ShaanxiPop,[0])
# There does not appear to be Shaanxi Population Data
#TianjinPopData = np.delete(TianjinPop,[0])
# There does not appear to be Tianjin Population Data
#InnerMongoliaPopData = np.delete(InnerMongoliaPop,[0])
# There does not appear to be Inner Mongolia Population Data
ShanghaiPopData = np.delete(ShanghaiPop,[0])
BeijingPopData = np.delete(BeijingPop,[0])
LiaoningPopData = np.delete(LiaoningPop,[0])
JilinPopData = np.delete(JilinPop,[0])
HeilongjiangPopData = np.delete(HeilongjiangPop,[0])
XinjiangPopData = np.delete(XinjiangPop,[0])
GansuPopData = np.delete(GansuPop,[0])
#TibetPopData = np.delete(TibetPop,[0])
# There does not appear to be Tibet Population Data
NingxiaPopData = np.delete(NingxiaPop,[0])
QinghaiPopData = np.delete(QinghaiPop,[0])

#%%
# Using this to make each year in the data
Time = np.linspace(1996, 2019, 24) #this would be my x data
# t_span = (1996,2019)

#%% Preliminaries prior to model validation.
"""
There is probably a faster way to evaluate each province, but for now this is the way I am doing it.
"""

# Using the average population total for our N
N = np.average(GuangdongPopData * 10000)

# Initial number of infected and recovered individuals, I0 and R0.
I0, R0 = GuangdongData[0], 0

# Initial number of Suscepitble , S0
S0 = N - I0 - R0

y0 = [S0, I0, R0] #just placing the intial values in matrix form
# Contact rate, beta, and mean recovery rate, gamma.
beta, gamma = 91.25/365 , 10/356
# average infection rate is 1-3 months in humans, death occurs 2-10 days after symptoms


#%%
# Setting parameters. Note that we can place bounds on values if need be.
params = Parameters()
params.add('beta', value = beta, min = 0, max = 10000) # contact rate
params.add('gamma', value = gamma, min = 0, max = 10000) # mean recovery rate
params.add('S0', value = S0, vary = False) # Initial number of susceptibles
params.add('I0', value = I0, vary = False) # Initial number of infectives
params.add('R0', value =R0, vary = False) # Initial number of recovered, typically 0

#%%
def sir(y, t, paras):
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

def g(t, x_0, paras):
    ''' Solution to the SIR ODE with initial condition S(0) = S0, I(0) = I0 '''

    ret = integrate.odeint(sir, x_0, t, args = (paras,))

    return ret

def residual(ps, ts, infdata):
    ''' Returns a residual, about which I'm not entirely sure. JTA '''
    y_0 = ps['S0'].value, ps['I0'].value, ps['R0'].value
    model = g(ts, y_0, ps)
    return ((model[:,1] - infdata).ravel(),) # (model[:, 0] - popdata).ravel()

def main():
    ''' This is the main function for testing and running this code. '''

    # plt.figure()
    # plt.scatter(Time, GuangdongData, marker = 'o', color = 'b', label = 'Number of Infected')
    #plt.scatter(Time, GuangdongPopData, marker = 'o', color = 'r', label = "Number of Susceptible/Population")

    # x0 = [S0, I0]
    # print(type(g(Time,x0, params)))
    # print(type(GuangxiData))

    # fit model
    results_infected = minimize(residual, params, args = (Time, GuangdongData), method = 'Nelder-Mead')
    # results_pop = minimize(residual, params, args = (Time, GuangdongData), method = 'Nelder-Mead')

    # check results of the fit
    data_fitted = GuangdongData + results_infected.residual.reshape(GuangdongData.shape)
    #pop_fitted = GuangxiPopData + results_pop.residual.reshape(GuangxiPopData.shape)

    # Plot the data on three separate curves for S(t), I(t) and R(t)
    fig = plt.figure(facecolor='w')
    ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
    #ax.plot(t, S, 'b', alpha=0.5, lw=2, label='Susceptible')
    #ax.plot(t, I, 'r', alpha=0.5, lw=2, label='Infected')
    #ax.plot(t, R, 'g', alpha=0.5, lw=2, label='Recovered with immunity')

    #ax.plot(Time, GuangdongPopData * 10000, 'o')
    ax.plot(Time, GuangdongData, 'o')
    ax.plot(Time, data_fitted, '-', color = 'k', linewidth=2)
    #ax.plot(Time, pop_fitted, '-', color = 'k', linewidth=2)

    ax.set_xlabel('Year')
    ax.set_ylabel('Number')
    #ax.set_ylim(0,15000)
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    ax.grid(visible=True, which='major', c='w', lw=2, ls='-')
    #legend = ax.legend()
    #legend.get_frame().set_alpha(0.5)
    #for spine in ('top', 'right', 'bottom', 'left'):
    #    ax.spines[spine].set_visible(False)
    plt.title("Guangdong")
    plt.show()

    #print(GuangdongPopData.shape)

    #display fitted statistics
    report_fit(results_infected)

if __name__ == '__main__':

    main()
