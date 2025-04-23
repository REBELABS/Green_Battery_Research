# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 18:16:20 2025

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

##Sample C Voltage analysis
#Carry out the Diffusion Kernel Density of which the Optimum will be arrived at
u_k, omega = diffKDE.KDE(sampleC_volt)

#Carry out the Diffusion KDE at the manual T
u_km, omegam = diffKDE.KDE(sampleC_volt, T = 0.00182)


#Carry out the Diffusion KDE for Optimum KDE * 2
u_km_double, omegam_double = diffKDE.KDE(sampleC_volt, T = 0.01038)

#Superimposed plotting-Ploting of histogram
counts, bins, patchs = plt.hist(sampleC_volt, bins = 'auto', density = True, alpha = 0.5, label = 'Histogram of Sample C Voltage')

#Plotting the KDEs
#Plotting KDE Optimum
plt.plot(omega, u_k, color = '#1f77b4ff', linestyle = '-', linewidth = 2, label = r"diffKDE, $T^*$")
#Plotting KDE Manual Optimum
plt.plot(omegam, u_km, color = 'green', linestyle = '--', linewidth = 2, label = r"diffKDE, $T^{m} = 0.00182$")
#Plotting for (KDE Optimum *2)
plt.plot(omegam_double, u_km_double, color = 'orange', linestyle = ':', linewidth = 2, label = r"diffKDE, $2{T^{*}}$")

#General Porperties of plot
plt.xlabel(r'Voltage $(V)$')
plt.ylabel('Density')
plt.grid(True)
plt.legend()

#Saving the histo
plt.tight_layout()
plt.savefig('histogram_plotsC.png', dpi = 300, bbox_inches = 'tight')
plt.close('histogram_plotsC.png')

#using the automati plots
#diffKDE.evol_plot(sampleC_volt)
#diffKDE.pilot_plot(sampleC_volt, T = 0.00182) #The pilot plot at the interest T
#diffKDE.custom_plot(sampleC_volt)

#Show plot
plt.tight_layout()
plt.show()


##Sample C Current analysis
#Carry out the Diffusion Kernel Density of which the Optimum will be arrived at
u_k, omega = diffKDE.KDE(sampleC_curr)

#Carry out the Diffusion KDE at the manual T
u_km, omegam = diffKDE.KDE(sampleC_curr, T = 0.01094)


#Carry out the Diffusion KDE for Optimum KDE * 2
u_km_double, omegam_double = diffKDE.KDE(sampleC_curr, T = 0.0843)

#Superimposed plotting-Ploting of histogram
counts, bins, patchs = plt.hist(sampleC_curr, bins = 10, density = True, alpha = 0.5, label = 'Histogram of Sample C Current')

#Plotting the KDEs
#Plotting KDE Optimum
plt.plot(omega, u_k, color = '#1f77b4ff', linestyle = '-', linewidth = 2, label = r"diffKDE, $T^*$")
#Plotting KDE Manual Optimum
plt.plot(omegam, u_km, color = 'green', linestyle = '--', linewidth = 2, label = r"diffKDE, $T^{m} = 0.01094$")
#Plotting for (KDE Optimum *2)
plt.plot(omegam_double, u_km_double, color = 'orange', linestyle = ':', linewidth = 2, label = r"diffKDE, $2{T^{*}}$")

#General Porperties of plot
plt.xlabel(r'Current $(A)$')
plt.ylabel('Density')
plt.grid(True)
plt.legend()

#Saving the histo
plt.tight_layout()
#plt.savefig('histogram_plotsCcurrent.png', dpi = 300, bbox_inches = 'tight')
#plt.close('histogram_plotsCcurrent.png')

#using the automati plots
diffKDE.evol_plot(sampleC_curr)
diffKDE.pilot_plot(sampleC_curr, T = 0.01094) #The pilot plot at the interest T
#diffKDE.custom_plot(sampleC_curr)

#Show plot
plt.tight_layout()
plt.show()
