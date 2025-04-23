# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 13:23:34 2025

@author: user
"""

import os
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from diffKDE import diffKDE
plt.rcParams['text.usetex'] = True

#print(dir(diffKDE)) #Use to get functions associated with it
#print(np.__version__)
#print(os.getcwd()) #Use to get the dir you are working at

#Load the Excel file into a DataFrame
df = pd.read_excel('DAV.xlsx', sheet_name = 'Sheet9', header = 1)

#print(df)
#print(df.columns)

#Save necessary VOLTAGE columns as data series into their variables
#SampleA
sampleA_volt = df.iloc[:,1].dropna().reset_index(drop = True)
#print(sampleA_volt)

 #For sample B, have to limit the column to be of same lenght as A
sampleB_volt = df.iloc[0:56,4].reset_index(drop=True)
#print(sampleB_volt)

 #For sample C, have to limit the column to be of same lenght as A
sampleC_volt = df.iloc[0:56,7].reset_index(drop=True)
#print(sampleC_volt)

 #Print Voltage samples side by side
full_volt_view = pd.concat([sampleA_volt, sampleB_volt, sampleC_volt], axis = 1)
full_volt_view.columns = ['SampleA Voltage', 'SampleB Voltage', 'SampleC Voltage']
#print(full_volt_view)

#Save necessary CURRENT columns as data series into their variables
#SampleB
sampleB_curr = df.iloc[0:32,10].reset_index(drop = True)
#print(sampleB_curr)

#For sample C, have to limit the column to be of same lenght as A
sampleC_curr = df.iloc[0:32,13].reset_index(drop=True)
#print(sampleC_curr)

#Print CURRENT samples side by side
full_curr_view = pd.concat([sampleB_curr, sampleC_curr], axis = 1)
full_curr_view.columns = ['SampleB Current', 'SampleC Current']
#print(full_curr_view)

#Converting to numpy array which is faster to use in maths
sampleA_volt = sampleA_volt.to_numpy()
sampleB_volt = sampleB_volt.to_numpy()
sampleC_volt = sampleC_volt.to_numpy()
sampleB_curr = sampleB_curr.to_numpy()
sampleC_curr = sampleC_curr.to_numpy()

##Sample A Voltage analysis
#Carry out the Diffusion Kernel Density of which the Optimum will be arrived at
u_k, omega = diffKDE.KDE(sampleA_volt)

#Carry out the Diffusion KDE at the manual T
#u_km, omegam = diffKDE.KDE(sampleA_volt, T = 0.00064)


#Carry out the Diffusion KDE for Optimum KDE * 2
#u_km_double, omegam_double = diffKDE.KDE(sampleA_volt, T = 0.00383)

#Adjusting the x-axis
#xticks = np.round(np.linspace(omega.min(), omega.max(), 12), 2)
#plt.locator_params(axis='x', nbins=10) as a above but just telling how many ticks

#Adjusting the x-axis
#xticksm = np.round(np.linspace(omegam.min(), omegam.max(), 12), 2)

#Superimposed plotting-Ploting of histogram
counts, bins, patchs = plt.hist(sampleA_volt, bins = 10, density = True, alpha = 0.5, label = 'Histogram of Sample A Voltage')

#Plotting the KDEs
#Plotting KDE Optimum
plt.plot(omega, u_k, color = '#1f77b4ff', linestyle = '-', linewidth = 2, label = r"diffKDE, $T^*$")
#Plotting KDE Manual Optimum
#plt.plot(omegam, u_km, color = 'green', linestyle = '--', linewidth = 2, label = r"diffKDE, $T^{m} = 0.00064$")
#Plotting for (KDE Optimum *2)
#plt.plot(omegam_double, u_km_double, color = 'orange', linestyle = ':', linewidth = 2, label = r"diffKDE, $2{T^{*}}$")

#General Porperties of plot
plt.xlabel(r'Voltage $(V)$')
plt.ylabel('Density')
plt.grid(True)
plt.legend()

#Saving the histo
plt.tight_layout()
#plt.savefig('histogram_plots.png', dpi = 300, bbox_inches = 'tight')
plt.close('histogram_plots.png')

#using the automati plots
#diffKDE.evol_plot(sampleA_volt)
#diffKDE.pilot_plot(sampleA_volt, T = 0.00064) #The pilot plot at the interest T
#diffKDE.custom_plot(sampleA_volt)

#Show plot
plt.tight_layout()
plt.show()




#Manual Ploting
#plt.plot(omega, u_k)
#plt.xticks(xticks)
#plt.title('Diffusion KDE of Sample A')
#plt.xlabel('Voltage (Volts)')
#plt.ylabel('Estimated Density')
#plt.grid(True)