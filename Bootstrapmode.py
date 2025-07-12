# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 23:22:15 2025
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 13:23:34 2025

@author: Dr. REBEL-ious
"""

import os
import numpy as np
import pandas as pd
import scipy
from scipy.signal import find_peaks, peak_widths
from multiprocessing import Pool
import matplotlib.pyplot as plt
from diffKDE import diffKDE
import time

plt.rcParams['text.usetex'] = True

#print(dir(diffKDE)) #Use to get functions associated with it
#print(np.__version__)
#print(os.getcwd()) #Use to get the dir you are working at

#Load the Excel file into a DataFrame
df = pd.read_excel(r'C:\Users\user\Desktop\Pyhton\py40\diffKDE\DAV.xlsx', sheet_name = 'Sheet9', header = 1)

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
#----------------------------------------------------------------------------------
#Bootstrap CI KDE comparison
#List of bootstrap modes voltage
#DEF statement for KDE and peak selection
def bootstrap_mode(sample):
    re_sample = np.random.choice(sample, size=len(sample), replace=True)
    re_u_k, re_omega = diffKDE.KDE(re_sample)
    re_sample_peaks, _ = find_peaks(re_u_k)
    if len(re_sample_peaks) > 0:
        re_sample_tall_index = re_sample_peaks[np.argmax(re_u_k[re_sample_peaks])]
        return re_omega[re_sample_tall_index]
    else:
        return None
        
if __name__ == "__main__": #Important to prevent looping unending and unintentional loops  when pip
    #Set iterations
    n_interations = 200
    for size in [2, 3, 5, 10, 15]:
        start = time.time()
        with Pool() as pool:
            bootstrap_sampA_volt_modes = list(filter(None,pool.map(bootstrap_mode, [sampleA_volt]*n_interations, chunksize=size)))
            bootstrap_sampB_volt_modes = list(filter(None,pool.map(bootstrap_mode, [sampleB_volt]*n_interations, chunksize=size)))
            bootstrap_sampC_volt_modes = list(filter(None,pool.map(bootstrap_mode, [sampleC_volt]*n_interations, chunksize=size)))
            print(f'Total time is {time.time()-start} for {size}')

