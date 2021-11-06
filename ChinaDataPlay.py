# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 13:50:08 2021

@author: chris
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# reading csv files. The excel read gave me some weird outputs. Just converted the .xlsx file to a .csv file. Won't work 
# with multiple sheet excel workbooks

#ChinaDataTest = pd.read_excel('1996-2020 rabies province.xlsx')
ChinaDataTest =  pd.read_csv('1996-2020 rabies province.csv')
ChinaPopTest = pd.read_excel('Chinese_population_data.xlsx')

#print(ChinaPop)

# fixing the data to be better readable

#takes the first row, which is the header
header =  ChinaDataTest.iloc[0]
#print(header)

#take the data, now minus the header
ChinaDataTest = ChinaDataTest[1:]
#print(ChinaDataTest)

# removing the rows with NaN values.
ChinaDataTest = ChinaDataTest.dropna(how = "all", axis = 0)
ChinaPopTest = ChinaPopTest.dropna(how = 'all', axis = 0)


# removing the columns with NaN values.
ChinaDataTest = ChinaDataTest.dropna(how = 'all', axis = 'columns')
ChinaPopTest = ChinaPopTest.dropna(how = 'all', axis = 'columns')

#print(ChinaPopTest)
#Renaming the column names
ChinaDataTest.columns=["Province",'1996','1997','1998', '1999', '2000', '2001', '2002', '2003',
                        '2004', '2005','2006', '2007', '2008', '2009', '2010', '2011', '2012', 
                        '2013', '2014', '2015' ,'2016', '2017', '2018', '2019', '2020']

ChinaPopTest.columns = ["Province",'1996','1997','1998', '1999', '2000', '2001', '2002', '2003',
                        '2004', '2005','2006', '2007', '2008', '2009', '2010', '2011', '2012', 
                        '2013', '2014', '2015' ,'2016', '2017', '2018', '2019']

# Renaming for convenience
cd = ChinaDataTest
cp = ChinaPopTest
#print(cd)
# removing the last column so that ChinaDataTest and ChinaPopTest are mathchin
cd.drop(columns = cd.columns[-1], axis = 1, inplace = True)
#print(cd)
#%%
# Getting the data for individual provinces, just need to change the name after ==
GuangxiInfected = cd.loc[cd['Province'] == 'Guangxi']
GuangxiPop = cp.loc[cp['Province'] == 'Guangxi']
#print(Guangxi)

# Converts the dataframe to a numpy array
GuangxiInfected = GuangxiInfected.to_numpy()
GuangxiPop = GuangxiPop.to_numpy()

# Removing the first element, i.e. the string 'Guangxi'
GuangxiInfected = np.delete(GuangxiInfected,[0])
GuangxiPop = np.delete(GuangxiPop,[0])

# Multiplying the population data by 10000 to get the actual value
GuangxiPop = GuangxiPop * 10000

# Using this to make each year in the data
Time = np.linspace(1996, 2019, 24)

# The same data and plot as the excel spreadsheet!!!
plt.plot(Time, GuangxiInfected)
plt.plot(Time, GuangxiPop)

