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

##Sample A Voltage analysis
#Carry out the Diffusion Kernel Density of which the Optimum will be arrived at
u_k, omega = diffKDE.KDE(sampleA_volt)

#Carry out the Diffusion KDE at the manual T
u_km, omegam = diffKDE.KDE(sampleA_volt, T = 0.00064)


#Carry out the Diffusion KDE for Optimum KDE * 2
u_km_double, omegam_double = diffKDE.KDE(sampleA_volt, T = 0.00383)

#Sample Bvoltage from the other file
u_kB, omegaB = diffKDE.KDE(sampleB_volt)
#Sample Cvoltage from the other file
u_kC, omegaC = diffKDE.KDE(sampleC_volt, T = 0.00182)
#Sample Bcurrent from the other file
u_kBC, omegaBC = diffKDE.KDE(sampleB_curr, T = 0.03369)
#Sample Ccurrent from the other file
u_kCC, omegaCC = diffKDE.KDE(sampleC_curr, T = 0.01094)


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
plt.plot(omegam, u_km, color = 'green', linestyle = '--', linewidth = 2, label = r"diffKDE, $T^{m} = 0.00064$")
#Plotting for (KDE Optimum *2)
plt.plot(omegam_double, u_km_double, color = 'orange', linestyle = ':', linewidth = 2, label = r"diffKDE, $2{T^{*}}$")

#General Porperties of plot
plt.xlabel(r'Voltage $(V)$')
plt.ylabel('Density')
plt.grid(True)
plt.legend()

#Saving the histo
plt.tight_layout()
plt.savefig('histogram_plotsA.png', dpi = 300, bbox_inches = 'tight')
plt.close('histogram_plotsA.png')

# #using the automati plots
# diffKDE.evol_plot(sampleA_volt)
# diffKDE.pilot_plot(sampleA_volt, T = 0.00064) #The pilot plot at the interest T
# diffKDE.custom_plot(sampleA_volt)

# #Show plot
# plt.tight_layout()
# plt.show()


#Manual Ploting
#plt.plot(omega, u_k)
#plt.xticks(xticks)
#plt.title('Diffusion KDE of Sample A')
#plt.xlabel('Voltage (Volts)')
#plt.ylabel('Estimated Density')
#plt.grid(True)

#Identifying the peaks for sample A-C voltage KDE
volt_peaks,_ = find_peaks(u_km)
volt_peaksB,_ = find_peaks(u_kB)
volt_peaksC,_ = find_peaks(u_kC)

#Estimating the voltage widths
result_half = peak_widths(u_km, volt_peaks, rel_height=0.25)
result_halfB = peak_widths(u_kB, volt_peaksB, rel_height=0.25)
result_halfC = peak_widths(u_kC, volt_peaksC, rel_height=0.25)

#Label the peaks
A=("SP1","SP2")
#Logic to turn of repeating legend of the width lines
label_logic=False

#Samples merging as a list of tuples
samples = [("Sample A", volt_peaks, omegam, u_km, "orange", result_half, "SAP","---",'-'),
           ("Sample B", volt_peaksB, omegaB, u_kB, "red", result_halfB, "SBP", "///",':'),
           ("Sample C", volt_peaksC, omegaC, u_kC, "green", result_halfC, "SCP", r"'\\\'",'--')]

#Unpacking the tuples and looping through individual item of tuple
for name, volt_peak, omegaa, u_kk, color, result_halff, label_prefix, hatch, linestyle in samples:
    peak_volt = omegaa[volt_peak]

#Getting the values from the array truple(result_half) containing width, height, left and right edges
    for i in range(len(result_halff[0])):
        height_volt= result_halff[1][i]
        left_edge= omegaa[int(result_halff[2][i])]
        right_edge= omegaa[int(result_halff[3][i])]
        width_volt= (right_edge) - (left_edge)
        #Shaded region under the peak for sample A voltage
        mask = (omegaa>=left_edge) & (omegaa<=right_edge)
        plt.fill_between(omegaa[mask], u_kk[mask], facecolor = 'none', edgecolor=color,hatch=hatch,linewidth=0.0, alpha=0.3)
        #FMCH line
        plt.hlines(height_volt, left_edge, right_edge, color="blue", label = 'FWCM' if not label_logic else '', linewidth = 1)
        #To stop repeating legend for widthline
        label_logic=True
        #Vertical edge lines at start and end of the width
        plt.axvline(left_edge, color=color,linestyle=':', linewidth = 0.8)
        plt.axvline(right_edge, color=color,linestyle=':', linewidth = 0.8)
#Label the peaks
    for i in range(len(volt_peaks)):
        x = peak_volt[i]
        y = u_kk[volt_peak[i]]
        label = f'{A[i]}'
        plt.text(x, y+0.3, label,fontsize=8, color=color,ha='center')
        print(f'{name} peak Voltage {i+1}: {x:.4f}')

#Composite plot of sampleA-C voltage KDE
    plt.plot(omegaa, u_kk, label = name, linewidth = 2, color = color, linestyle = linestyle)    

plt.xlabel(r'Voltage $(V)$')
plt.ylabel('Density')
plt.title(r"\textbf{Selective Shading of Voltage KDE Regions}",fontsize=14,weight='bold')
plt.grid(alpha=0.4)
#Reordering the legend so the samples come first
old_plot_objts, old_legend_txt = plt.gca().get_legend_handles_labels()
new_order = ["Sample A", "Sample B", "Sample C", "FWCM"]
new_plot_objts = [old_plot_objts[old_legend_txt.index(label_name)] for label_name in new_order]
new_legend_txt = new_order
plt.legend(new_plot_objts, new_legend_txt)
plt.ylim(bottom=0)
plt.tight_layout()
plt.savefig('ModesAC.png', dpi = 300, bbox_inches = 'tight')
plt.show()

#-------------------------------------------------------------------------------
#Identifying the peaks for sample B-C current KDE
curr_peaksBC,_ = find_peaks(u_kBC)
curr_peaksCc,_ = find_peaks(u_kCC)
curr_peaksCC = [curr_peaksCc[1]]
print(curr_peaksCC)

#Estimating the current widths
result_halfBC = peak_widths(u_kBC, curr_peaksBC, rel_height=0.25)
result_halfCC = peak_widths(u_kCC, curr_peaksCC, rel_height=0.25)
print(result_halfCC)

#Logic to turn of repeating legend of the width lines
label_logic=False

#Samples merging as a list of tuples
samplesC = [("Sample B", curr_peaksBC, omegaBC, u_kBC, "red", result_halfBC, "SBP", "///",':'),
           ("Sample C", curr_peaksCC, omegaCC, u_kCC, "green", result_halfCC, "SCP", r"'\\\'",'--')]

#Unpacking the tuples and looping through individual item of tuple
for nameC, curr_peak, omegaaC, u_kkC, colorC, result_halffC, label_prefixC, hatchC, linestyleC in samplesC:
    peak_curr = omegaaC[curr_peak]
    print(peak_curr)

#Getting the values from the array truple(result_half) containing width, height, left and right edges
    for i in range(len(result_halffC[0])):
        height_voltC= result_halffC[1][i]
        left_edgeC= omegaaC[int(result_halffC[2][i])]
        right_edgeC= omegaaC[int(result_halffC[3][i])]
        width_voltC= (right_edgeC) - (left_edgeC)
        #Shaded region under the peak for sample A voltage
        maskC = (omegaaC>=left_edgeC) & (omegaaC<=right_edgeC)
        plt.fill_between(omegaaC[maskC], u_kkC[maskC], facecolor = 'none', edgecolor=colorC,hatch=hatchC,linewidth=0.0, alpha=0.3)
        #FMCH line
        plt.hlines(height_voltC, left_edgeC, right_edgeC, color="blue", label = 'FWCM' if not label_logic else '', linewidth = 1)
        #To stop repeating legend for widthline
        label_logic=True
        #Vertical edge lines at start and end of the width
        plt.axvline(left_edgeC, color=colorC,linestyle=':', linewidth = 0.8)
        plt.axvline(right_edgeC, color=colorC,linestyle=':', linewidth = 0.8)
#Manual setting for sample C SP1
    left_edgeCC = 0.35702
    right_edgeCC = 0.40116
    heightCC = 2.1508
    maskCC = (omegaaC>=left_edgeCC) & (omegaaC<=right_edgeCC)
    plt.fill_between(omegaaC[maskCC], u_kkC[maskCC], facecolor = 'none', edgecolor=colorC,hatch=hatchC,linewidth=0.0, alpha=0.3)
    plt.hlines(heightCC, left_edgeCC, right_edgeCC, color="blue", linewidth = 1)
    plt.axvline(left_edgeCC, color=colorC,linestyle=':', linewidth = 0.8)
    plt.axvline(right_edgeCC, color=colorC,linestyle=':', linewidth = 0.8)
#Composite plot of sampleA-C voltage KDE    
    plt.plot(omegaaC, u_kkC, label = nameC, linewidth = 2, color = colorC, linestyle = linestyleC)    

#Label the peaks
labels = [("SP1"), ("SP1"), ("SP2")]
x = [(0.4769), (0.3786), (0.4933)]
y = [(1.341), (2.195), (2.178)]
colorCC = [("red"), ("green"), ("green")]
for i in range(len(x)):
    xx = x[i]  
    yy = y[i]
    label = f'{labels[i]}'
    plt.text(xx, yy, label,fontsize=8, color=colorCC[i],ha='center')

plt.xlabel(r'Current $(A)$')
plt.ylabel('Density')
plt.title(r"\textbf{Selective Shading of Current KDE Regions}",fontsize=14,weight='bold')
plt.grid(alpha=0.4)
#Reordering the legend so the samples come first
old_plot_objtsC, old_legend_txtC = plt.gca().get_legend_handles_labels()
new_orderC = ["Sample B", "Sample C", "FWCM"]
new_plot_objtsC = [old_plot_objtsC[old_legend_txtC.index(label_nameC)] for label_nameC in new_orderC]
new_legend_txtC = new_orderC
plt.legend(new_plot_objtsC, new_legend_txtC)
plt.ylim(bottom=0)
plt.tight_layout()
plt.savefig('ModesBC.png', dpi = 300, bbox_inches = 'tight')
plt.show()

#-------------------------------------------------------------------------------    
start = time.time()
#Bootstrap CI KDE comparison
n_interations = 20000

#List of bootstrap modes voltage
bootstrap_sampA_volt_modes = []
bootstrap_sampB_volt_modes = []
bootstrap_sampC_volt_modes = []

#Bootstrap for Sample A voltage
for _ in range(n_interations):
    resampleA_volt = np.random.choice(sampleA_volt, size=len(sampleA_volt), replace=True)
    resampleA_volt_u_k, resampleA_volt_omega = diffKDE.KDE(resampleA_volt)
    resampleA_volt_peaks,_ = find_peaks(resampleA_volt_u_k)
    if len(resampleA_volt_peaks) > 0:
        resampleA_tallest_peak_index = resampleA_volt_peaks[np.argmax(resampleA_volt_u_k[resampleA_volt_peaks])]
        bootstrap_sampA_volt_modes.append(resampleA_volt_omega[resampleA_tallest_peak_index])
  
#Bootstrap for Sample B voltage
for _ in range(n_interations):
    resampleB_volt = np.random.choice(sampleB_volt, size=len(sampleB_volt), replace=True)
    resampleB_volt_u_k, resampleB_volt_omega = diffKDE.KDE(resampleB_volt)
    resampleB_volt_peaks,_ = find_peaks(resampleB_volt_u_k)
    if len(resampleB_volt_peaks) > 0:
        resampleB_tallest_peak_index = resampleB_volt_peaks[np.argmax(resampleB_volt_u_k[resampleB_volt_peaks])]
        bootstrap_sampB_volt_modes.append(resampleB_volt_omega[resampleB_tallest_peak_index])

#Bootstrap for Sample C voltage
for _ in range(n_interations):
    resampleC_volt = np.random.choice(sampleC_volt, size=len(sampleC_volt), replace=True)
    resampleC_volt_u_k, resampleC_volt_omega = diffKDE.KDE(resampleC_volt)
    resampleC_volt_peaks,_ = find_peaks(resampleC_volt_u_k)
    if len(resampleC_volt_peaks) > 0:
        resampleC_tallest_peak_index = resampleC_volt_peaks[np.argmax(resampleC_volt_u_k[resampleC_volt_peaks])]
        bootstrap_sampC_volt_modes.append(resampleC_volt_omega[resampleC_tallest_peak_index])
#The total time for boostrapmode
print(f'Total time is {time.time()-start}')
#-----------------------------------------------------------------------------------------------

#Saving the Mode Bootstrap Voltage results
#First convert to numpy for fast computations
bootstrap_sampA_volt_modes = np.array(bootstrap_sampA_volt_modes)
bootstrap_sampB_volt_modes = np.array(bootstrap_sampB_volt_modes)
bootstrap_sampC_volt_modes = np.array(bootstrap_sampC_volt_modes)

#Save to npy and csv file
np.savez("Voltage Bootstrap Modes.npz",Sample_A=bootstrap_sampA_volt_modes,
         Sample_B=bootstrap_sampB_volt_modes,
         Sample_C=bootstrap_sampC_volt_modes)
#Padded with NaN to be of the same enght to avoid error
boot_mode_volt = pd.DataFrame({"Sample A Bootstrap Mode":pd.Series(bootstrap_sampA_volt_modes),
                               "Sample B Bootstrap Mode":pd.Series(bootstrap_sampB_volt_modes),
                               "Sample C Bootstrap Mode":pd.Series(bootstrap_sampC_volt_modes)})
boot_mode_volt.to_csv("Voltage Bootstrap Modes.csv", index=False)
#-------------------------------------------------------------------------------------------------

#Count number of peaks out of 20,000 simulations
peak_counts_sampA = len(bootstrap_sampA_volt_modes)
peak_counts_sampB = len(bootstrap_sampB_volt_modes)
peak_counts_sampC = len(bootstrap_sampC_volt_modes)
print(peak_counts_sampA, peak_counts_sampB, peak_counts_sampC)
#Save the output to a log .txt file
with open("Volt_Bootstrap_logs.txt","w") as f:
    f.write("#Counts number of peaks from 20,000 simulations\n")
    f.write(f"Sample A Voltage Peaks: {peak_counts_sampA}\n")
    f.write(f"Sample B Voltage Peaks: {peak_counts_sampB}\n")
    f.write(f"Sample C Voltage Peaks: {peak_counts_sampC}\n")
#----------------------------------------------------------------------------------------------------

#Voltage Mode Confidence Interval
mode_ci_A = np.percentile(bootstrap_sampA_volt_modes, [2.5, 97.5])
mode_ci_B = np.percentile(bootstrap_sampB_volt_modes, [2.5, 97.5])
mode_ci_C = np.percentile(bootstrap_sampC_volt_modes, [2.5, 97.5])
print(mode_ci_A, mode_ci_B, mode_ci_C)
#---------------------------------------------------------------------------------------------------------

##Sample A Voltage total Sum
total_sampA_volt = np.sum((bootstrap_sampA_volt_modes >= mode_ci_A[0]) & 
                          (bootstrap_sampA_volt_modes <= mode_ci_A[1]))
##Divide the Bootstrap Modes into classes
#Sample A Voltage
ci_sampA_volt_low = np.sum((bootstrap_sampA_volt_modes >= mode_ci_A[0])
                & (bootstrap_sampA_volt_modes <= 0.07483))
ci_sampA_volt_medium = np.sum((bootstrap_sampA_volt_modes > 0.07483)
                & (bootstrap_sampA_volt_modes <= 0.08707))
ci_sampA_volt_high = np.sum((bootstrap_sampA_volt_modes > 0.08707)
                & (bootstrap_sampA_volt_modes <= mode_ci_A[1]))
#Percentage number of peaks in low
per_sampA_l = (ci_sampA_volt_low/total_sampA_volt)*100
per_sampA_m = (ci_sampA_volt_medium/total_sampA_volt)*100
per_sampA_h = (ci_sampA_volt_high/total_sampA_volt)*100
perA_all = (per_sampA_l,per_sampA_m, per_sampA_h)
#Robust typical value (median)
median_sampA_l = np.median(bootstrap_sampA_volt_modes[(bootstrap_sampA_volt_modes >= mode_ci_A[0])
                & (bootstrap_sampA_volt_modes <= 0.07483)])
median_sampA_m = np.median(bootstrap_sampA_volt_modes[(bootstrap_sampA_volt_modes > 0.07483)
                & (bootstrap_sampA_volt_modes <= 0.08707)])
median_sampA_h = np.median(bootstrap_sampA_volt_modes[(bootstrap_sampA_volt_modes > 0.08707)
                & (bootstrap_sampA_volt_modes <= mode_ci_A[1])])
medianA_all = (median_sampA_l,median_sampA_m,median_sampA_h)

#Append into the bootstrap log .txt file
with open("Volt_Bootstrap_logs.txt","a") as f:
    f.write("\n--------------------\n")
    f.write("#Sample A Voltage Bootstrap Details\n")
    f.write(f"CI: Lower Bound is {mode_ci_A[0]:.4f}V, Upper Bound is {mode_ci_A[1]:.4f}V")
    f.write("\n")
    f.write("Classification of the Voltage Bootstrap Modes\n")
    f.write(f"Total Voltage Peaks within CI: {total_sampA_volt:.0f}\n")
    f.write(f"Low Class ({mode_ci_A[0]:.4f}V - 0.07483V): {ci_sampA_volt_low:.0f}; Percentage: {per_sampA_l:.4f}%; Median: {median_sampA_l:.4f}V\n")
    f.write(f"Medium Class (> (0.07483)V - 0.08707V): {ci_sampA_volt_medium:.0f}; Percentage: {per_sampA_m:.4f}%; Median: {median_sampA_m:.4f}V\n")
    f.write(f"High Class (0.08707V - {mode_ci_A[1]:.4f}V): {ci_sampA_volt_high:.0f}; Percentage: {per_sampA_h:.4f}%; Median: {median_sampA_h:.4f}V\n")  
#------------------------------------------------------------------------------------------
##Sample B Voltage total Sum
total_sampB_volt = np.sum((bootstrap_sampB_volt_modes >= mode_ci_B[0]) & 
                          (bootstrap_sampB_volt_modes <= mode_ci_B[1]))
##Divide the Bootstrap Modes into classes
#Sample B Voltage
ci_sampB_volt_low = np.sum((bootstrap_sampB_volt_modes >= mode_ci_B[0])
                & (bootstrap_sampB_volt_modes <= 0.10256))
ci_sampB_volt_medium = np.sum((bootstrap_sampB_volt_modes > 0.10256)
                & (bootstrap_sampB_volt_modes <= 0.12316))
ci_sampB_volt_high = np.sum((bootstrap_sampB_volt_modes > 0.12316)
                & (bootstrap_sampB_volt_modes <= mode_ci_B[1]))
#Percentage number of peaks in low
per_sampB_l = (ci_sampB_volt_low/total_sampB_volt)*100
per_sampB_m = (ci_sampB_volt_medium/total_sampB_volt)*100
per_sampB_h = (ci_sampB_volt_high/total_sampB_volt)*100
perB_all = (per_sampB_l,per_sampB_m, per_sampB_h)
#Robust typical value (median)
median_sampB_l = np.median(bootstrap_sampB_volt_modes[(bootstrap_sampB_volt_modes >= mode_ci_B[0])
                & (bootstrap_sampB_volt_modes <= 0.10256)])
median_sampB_m = np.median(bootstrap_sampB_volt_modes[(bootstrap_sampB_volt_modes > 0.10256)
                & (bootstrap_sampB_volt_modes <= 0.12316)])
median_sampB_h = np.median(bootstrap_sampB_volt_modes[(bootstrap_sampB_volt_modes > 0.12316)
                & (bootstrap_sampB_volt_modes <= mode_ci_B[1])])
medianB_all = (median_sampB_l,median_sampB_m,median_sampB_h)

#Append into the bootstrap log .txt file
with open("Volt_Bootstrap_logs.txt","a") as f:
    f.write("\n--------------------\n")
    f.write("#Sample B Voltage Bootstrap Details\n")
    f.write(f"CI: Lower Bound is {mode_ci_B[0]:.4f}V, Upper Bound is {mode_ci_B[1]:.4f}V")
    f.write("\n")
    f.write("Classification of the Voltage Bootstrap Modes\n")
    f.write(f"Total Voltage Peaks within CI: {total_sampB_volt:.0f}\n")
    f.write(f"Low Class ({mode_ci_B[0]:.4f}V - 0.10256V): {ci_sampB_volt_low:.0f}; Percentage: {per_sampB_l:.4f}%; Median: {median_sampB_l:.4f}V\n")
    f.write(f"Medium Class (> (0.10256)V - 0.12316V): {ci_sampB_volt_medium:.0f}; Percentage: {per_sampB_m:.4f}%; Median: {median_sampB_m:.4f}V\n")
    f.write(f"High Class (0.12316V - {mode_ci_B[1]:.4f}V): {ci_sampB_volt_high:.0f}; Percentage: {per_sampB_h:.4f}%; Median: {median_sampB_h:.4f}V\n")  
#------------------------------------------------------------------------------------------

##Sample C Voltage total Sum
total_sampC_volt = np.sum((bootstrap_sampC_volt_modes >= mode_ci_C[0]) & 
                          (bootstrap_sampC_volt_modes <= mode_ci_C[1]))
##Divide the Bootstrap Modes into classes
#Sample C Voltage
ci_sampC_volt_low = np.sum((bootstrap_sampC_volt_modes >= mode_ci_C[0])
                & (bootstrap_sampC_volt_modes <= 0.12543))
ci_sampC_volt_medium = np.sum((bootstrap_sampC_volt_modes > 0.12543)
                & (bootstrap_sampC_volt_modes <= 0.15294))
ci_sampC_volt_high = np.sum((bootstrap_sampC_volt_modes > 0.15294)
                & (bootstrap_sampC_volt_modes <= mode_ci_C[1]))
#Percentage number of peaks in low
per_sampC_l = (ci_sampC_volt_low/total_sampC_volt)*100
per_sampC_m = (ci_sampC_volt_medium/total_sampC_volt)*100
per_sampC_h = (ci_sampC_volt_high/total_sampC_volt)*100
perC_all = (per_sampC_l,per_sampC_m, per_sampC_h)

#Robust typical value (median)
median_sampC_l = np.median(bootstrap_sampC_volt_modes[(bootstrap_sampC_volt_modes >= mode_ci_C[0])
                & (bootstrap_sampC_volt_modes <= 0.12543)])
median_sampC_m = np.median(bootstrap_sampC_volt_modes[(bootstrap_sampC_volt_modes > 0.12543)
                & (bootstrap_sampC_volt_modes <= 0.15294)])
median_sampC_h = np.median(bootstrap_sampC_volt_modes[(bootstrap_sampC_volt_modes > 0.15294)
                & (bootstrap_sampC_volt_modes <= mode_ci_C[1])])
medianC_all = (median_sampC_l,median_sampC_m,median_sampC_h)
#Append into the bootstrap log .txt file
with open("Volt_Bootstrap_logs.txt","a") as f:
    f.write("\n--------------------\n")
    f.write("#Sample C Voltage Bootstrap Details\n")
    f.write(f"CI: Lower Bound is {mode_ci_C[0]:.4f}V, Upper Bound is {mode_ci_C[1]:.4f}V")
    f.write("\n")
    f.write("Classification of the Voltage Bootstrap Modes\n")
    f.write(f"Total Voltage Peaks within CI: {total_sampC_volt:.0f}\n")
    f.write(f"Low Class ({mode_ci_C[0]:.4f}V - 0.12543V): {ci_sampC_volt_low:.0f}; Percentage: {per_sampC_l:.4f}%; Median: {median_sampC_l:.4f}V\n")
    f.write(f"Medium Class (> (0.12543)V - 0.15294V): {ci_sampC_volt_medium:.0f}; Percentage: {per_sampC_m:.4f}%; Median: {median_sampC_m:.4f}V\n")
    f.write(f"High Class (0.15294V - {mode_ci_C[1]:.4f}V): {ci_sampC_volt_high:.0f}; Percentage: {per_sampC_h:.4f}%; Median: {median_sampC_h:.4f}V\n")  
#------------------------------------------------------------------------------------------
##Bootstrap voltage comparison infor for BarChart
#List of infor for each sample
sampA_counts= [per_sampA_l,per_sampA_m,per_sampA_h]
sampB_counts= [per_sampB_l,per_sampB_m,per_sampB_h]
sampC_counts= [per_sampC_l,per_sampC_m,per_sampC_h]

#Labels and bar positions
labels = ['Low','Medium','High']
width_diff = 0.25
x_positions= np.arange(len(labels))

#-------------------------------------------------------------------------------------------------
#Mode CI Bootstrap Plots
mode_fig, mode_axs = plt.subplots(2,2, figsize=(12,8), constrained_layout = True)

#Create over-reaching tiltle for all plots
mode_fig.suptitle(r"\textbf{Bootstrap Mode Confidence Interval for Samples A-C Voltage}", fontsize = 14, weight = 'bold')

#For sample A Voltage
mode_axs[0,0].hist(bootstrap_sampA_volt_modes, bins='auto',alpha = 0.5)
mode_axs[0,0].axvline(mode_ci_A[0],color='red',linestyle='--',label=f'Lower CI: {mode_ci_A[0]:.3f}')
mode_axs[0,0].axvline(mode_ci_A[1],color='green',linestyle='--',label=f'Upper CI: {mode_ci_A[1]:.3f}')
mode_axs[0,0].set_title('(a) Sample A Voltage')
mode_axs[0,0].grid(alpha=0.4)
mode_axs[0,0].legend()
mode_axs[0,0].set_xlabel(r"Voltage $(V)$")
mode_axs[0,0].set_ylabel('Frequency')

#For sample B Voltage
mode_axs[0,1].hist(bootstrap_sampB_volt_modes, bins='auto',alpha = 0.5)
mode_axs[0,1].axvline(mode_ci_B[0],color='red',linestyle='--',label=f'Lower CI: {mode_ci_B[0]:.3f}')
mode_axs[0,1].axvline(mode_ci_B[1],color='green',linestyle='--',label=f'Upper CI: {mode_ci_B[1]:.3f}')
mode_axs[0,1].set_title('(b) Sample B Voltage')
mode_axs[0,1].grid(alpha=0.4)
mode_axs[0,1].legend()
mode_axs[0,1].set_xlabel(r"Voltage $(V)$")
mode_axs[0,1].set_ylabel('Frequency')

#For sample C Voltage
mode_axs[1,0].hist(bootstrap_sampC_volt_modes, bins='auto',alpha = 0.5)
mode_axs[1,0].axvline(mode_ci_C[0],color='red',linestyle='--',label=f'Lower CI: {mode_ci_C[0]:.3f}')
mode_axs[1,0].axvline(mode_ci_C[1],color='green',linestyle='--',label=f'Upper CI: {mode_ci_C[1]:.3f}')
mode_axs[1,0].set_title('(c) Sample C Voltage')
mode_axs[1,0].grid(alpha=0.4)
mode_axs[1,0].legend()
mode_axs[1,0].set_xlabel(r"Voltage $(V)$")
mode_axs[1,0].set_ylabel('Frequency')

#Bar chart for comparison of classifications
mode_axs[1,1].bar(x_positions-width_diff,sampA_counts,width_diff,label='Sample A', color='orange')
mode_axs[1,1].bar(x_positions,sampB_counts,width_diff,label='Sample B',color='red')
mode_axs[1,1].bar(x_positions+width_diff,sampC_counts,width_diff,label='Sample C',color='green')
#Barchart properties
mode_axs[1,1].set_ylim(0,80)
mode_axs[1,1].set_xticks(x_positions)
mode_axs[1,1].set_xticklabels(labels)
mode_axs[1,1].set_title('(d) Voltage Classification Across Samples')
mode_axs[1,1].set_ylabel('Percentage')
mode_axs[1,1].grid(alpha=0.2)
mode_axs[1,1].legend()
#Adding the median text on each bar
for i, (perc,median_value) in enumerate(zip(perA_all,medianA_all)):
    mode_axs[1,1].text(x_positions[i]-width_diff,perc+0.3,fr"${median_value:.4f}\,\mathrm{{V^*}}$",ha='center',va='bottom',fontsize=7)
for i, (perc,median_value) in enumerate(zip(perB_all,medianB_all)):
    mode_axs[1,1].text(x_positions[i],perc+0.3,fr"${median_value:.4f}\,\mathrm{{V^*}}$",ha='center',va='bottom',fontsize=7)
for i, (perc,median_value) in enumerate(zip(perC_all,medianC_all)):
    mode_axs[1,1].text(x_positions[i]+width_diff,perc+0.3,fr"${median_value:.4f}\,\mathrm{{V^*}}$",ha='center',va='bottom',fontsize=7)
plt.savefig("BootstrapMode.png", dpi = 300, bbox_inches = 'tight')
plt.show()

#-------------------------------------------------------------------------------------------------------

##Bootstrap Voltage Median 
n_interations = 20000
start = time.time()
#List of bootstrap median voltage
bootstrap_sampA_volt_median = []
bootstrap_sampB_volt_median = []
bootstrap_sampC_volt_median = []

#A DEF statement to run for all samples, returns the median for each
def bootstrap_median(data,n_iterations):
    return [np.median(np.random.choice(data,size=len(data),replace=True)) for _ in range(n_iterations)]

#Bootstrap for Sample A median
bootstrap_sampA_volt_median = bootstrap_median(sampleA_volt,n_interations)
#Bootstrap for Sample B median
bootstrap_sampB_volt_median = bootstrap_median(sampleB_volt,n_interations)
#Bootstrap for Sample C median
bootstrap_sampC_volt_median = bootstrap_median(sampleC_volt,n_interations)
     
#The total time for boostrapmedian
print(f'Total time is {time.time()-start}')
#-----------------------------------------------------------------------------------------------

#Saving the Median Bootstrap Voltage results
#First convert to numpy for fast computations
bootstrap_sampA_volt_median = np.array(bootstrap_sampA_volt_median)
bootstrap_sampB_volt_median = np.array(bootstrap_sampB_volt_median)
bootstrap_sampC_volt_median = np.array(bootstrap_sampC_volt_median)

#Save to npy and csv file
np.savez("Voltage Bootstrap Median.npz",Sample_A=bootstrap_sampA_volt_median,
         Sample_B=bootstrap_sampB_volt_median,
         Sample_C=bootstrap_sampC_volt_median)
#Padded with NaN to be of the same enght to avoid error
boot_median_volt = pd.DataFrame({"Sample A Bootstrap Median":pd.Series(bootstrap_sampA_volt_median),
                               "Sample B Bootstrap Median":pd.Series(bootstrap_sampB_volt_median),
                               "Sample C Bootstrap Median":pd.Series(bootstrap_sampC_volt_median)})
boot_median_volt.to_csv("Voltage Bootstrap Median.csv", index=False)
#-------------------------------------------------------------------------------------------------

#Count number of medians returned out of 20,000 simulations
median_counts_sampA = len(bootstrap_sampA_volt_median)
median_counts_sampB = len(bootstrap_sampB_volt_median)
median_counts_sampC = len(bootstrap_sampC_volt_median)
print(median_counts_sampA, median_counts_sampB, median_counts_sampC)

#----------------------------------------------------------------------------------------------------

#Voltage Median Confidence Interval
median_ci_A = np.percentile(bootstrap_sampA_volt_median, [2.5, 97.5])
median_ci_B = np.percentile(bootstrap_sampB_volt_median, [2.5, 97.5])
median_ci_C = np.percentile(bootstrap_sampC_volt_median, [2.5, 97.5])
print(median_ci_A, median_ci_B, median_ci_C)
#---------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
##Median CI Bootstrap Plots
median_fig, median_axs = plt.subplots(2,2, figsize=(12,8), constrained_layout = True)

#Create over-reaching tiltle for all plots
median_fig.suptitle(r"\textbf{Bootstrap Median Confidence Interval for Samples A-C Voltage}", fontsize = 14, weight = 'bold')

#Switch off the unsed subplot
median_axs[1,1].axis("off")
#For sample A Voltage
median_axs[0,0].hist(bootstrap_sampA_volt_median, bins='auto',alpha = 0.5)
median_axs[0,0].axvline(median_ci_A[0],color='red',linestyle='--',label=f'Lower CI: {median_ci_A[0]:.3f}')
median_axs[0,0].axvline(median_ci_A[1],color='green',linestyle='--',label=f'Upper CI: {median_ci_A[1]:.3f}')
median_axs[0,0].set_title('(a) Sample A Voltage')
median_axs[0,0].grid(alpha=0.4)
median_axs[0,0].legend()
median_axs[0,0].set_xlabel(r"Voltage $(V)$")
median_axs[0,0].set_ylabel('Frequency')

#For sample B Voltage
median_axs[0,1].hist(bootstrap_sampB_volt_median, bins='auto',alpha = 0.5)
median_axs[0,1].axvline(median_ci_B[0],color='red',linestyle='--',label=f'Lower CI: {median_ci_B[0]:.3f}')
median_axs[0,1].axvline(median_ci_B[1],color='green',linestyle='--',label=f'Upper CI: {median_ci_B[1]:.3f}')
median_axs[0,1].set_title('(b) Sample B Voltage')
median_axs[0,1].grid(alpha=0.4)
median_axs[0,1].legend(loc=left_edge)
median_axs[0,1].set_xlabel(r"Voltage $(V)$")
median_axs[0,1].set_ylabel('Frequency')

#For sample C Voltage
median_axs[1,0].hist(bootstrap_sampC_volt_median, bins='auto',alpha = 0.5)
median_axs[1,0].axvline(median_ci_C[0],color='red',linestyle='--',label=f'Lower CI: {median_ci_C[0]:.3f}')
median_axs[1,0].axvline(median_ci_C[1],color='green',linestyle='--',label=f'Upper CI: {median_ci_C[1]:.3f}')
median_axs[1,0].set_title('(c) Sample C Voltage')
median_axs[1,0].grid(alpha=0.4)
median_axs[1,0].legend()
median_axs[1,0].set_xlabel(r"Voltage $(V)$")
median_axs[1,0].set_ylabel('Frequency')
plt.savefig("BootstrapMedian.png", dpi = 300, bbox_inches = 'tight')
plt.show()

#-----------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
##Bootstrap Current CI KDE comparison
n_interations = 20000

#List of bootstrap modes current
bootstrap_sampB_curr = []
bootstrap_sampC_curr= []
bootstrap_sampB_curr_modes = []
bootstrap_sampC_curr_modes = []
#Properties for later diagnotics
failed_diffB_curr = 0 #To count the number of failed diffKDEs
failed_diffC_curr = 0
failed_sampleB_curr=[] #Capture the fialed samples to diagnostics later
failed_sampleC_curr=[]
start = time.time() #Current time. Purpose is to subtract from the "current time" when the simulation ends
#Bootstrap for Sample B current
for i in range(n_interations):
    resampleB_curr = np.random.choice(sampleB_curr, size=len(sampleB_curr), replace=True)
    bootstrap_sampB_curr.append(resampleB_curr)
    #Try the diffKDE if failure, skip and continue to rest
    try:
        resampleB_curr_u_k, resampleB_curr_omega = diffKDE.KDE(resampleB_curr)
    except Exception as e:
        print(f"Sample :{i} suffered: {e}")
        failed_sampleB_curr.append(resampleB_curr)
        failed_diffB_curr +=1
        continue
    resampleB_curr_peaks,_ = find_peaks(resampleB_curr_u_k)
    if len(resampleB_curr_peaks) > 0:
        resampleB_curr_tallest_peak_index = resampleB_curr_peaks[np.argmax(resampleB_curr_u_k[resampleB_curr_peaks])]
        bootstrap_sampB_curr_modes.append(resampleB_curr_omega[resampleB_curr_tallest_peak_index])
  
#Bootstrap for Sample C current
for j in range(n_interations):
    resampleC_curr = np.random.choice(sampleC_curr, size=len(sampleC_curr), replace=True)
    bootstrap_sampC_curr.append(resampleC_curr)
    try:
        resampleC_curr_u_k, resampleC_curr_omega = diffKDE.KDE(resampleC_curr)
    except Exception as e:
        print(f"Sample no:{j} suffered: {e}")
        failed_sampleC_curr.append(resampleC_curr)
        failed_diffC_curr +=1
        continue
    resampleC_curr_peaks,_ = find_peaks(resampleC_curr_u_k)
    if len(resampleC_curr_peaks) > 0:
        resampleC_curr_tallest_peak_index = resampleC_curr_peaks[np.argmax(resampleC_curr_u_k[resampleC_curr_peaks])]
        bootstrap_sampC_curr_modes.append(resampleC_curr_omega[resampleC_curr_tallest_peak_index])
print(len(failed_sampleB_curr))
#The total time for boostrapcurrent
print(f'Total time is {time.time()-start}')

#----------------------------------------------------------------------
#Store the resampled currents both csv and compress Numpy version
resampleCurr_df=pd.DataFrame({'Sample_B': bootstrap_sampB_curr,
                              'Sample_C': bootstrap_sampC_curr})
resampleCurr_df.to_csv('Resampled_Currents.csv',index=False)
np.savez('Resampled_Currents.npz',Sample_B=bootstrap_sampB_curr,
         Sample_C=bootstrap_sampC_curr)

##Saving the Mode Bootstrap Current results
#First convert to numpy for fast computations
bootstrap_sampB_curr_modes = np.array(bootstrap_sampB_curr_modes)
bootstrap_sampC_curr_modes = np.array(bootstrap_sampC_curr_modes)
#Save to npy and csv file
np.savez("Current_Bootstrap_Modes.npz",Sample_B=bootstrap_sampB_curr_modes,
         Sample_C=bootstrap_sampC_curr_modes)
#Padded with NaN to be of the same lenght to avoid error
boot_mode_curr = pd.DataFrame({"Sample_B_Bootstrap_Curr":pd.Series(bootstrap_sampB_curr_modes),
                               "Sample_C_Bootstrap_Curr":pd.Series(bootstrap_sampC_curr_modes)})
boot_mode_curr.to_csv("Current_Bootstrap_Modes.csv", index=False)

#Store the failed resampled data
failed_df = pd.DataFrame({'Sample_B':pd.Series(failed_sampleB_curr),
                          'Sample_C':pd.Series(failed_sampleC_curr)})
failed_df.to_csv('Failed_Current_Samples.csv',index=False)
np.savez('Failed_Current_Samples.npz', Sample_B=failed_sampleB_curr,
         sample_C=failed_sampleC_curr)

#-----------------------------------------------------------------------------------------------

#Count number of peaks out of 20,000 simulations
peak_counts_sampB_curr = len(bootstrap_sampB_curr_modes)
peak_counts_sampC_curr = len(bootstrap_sampC_curr_modes)
print(peak_counts_sampB_curr, peak_counts_sampC_curr)
#Save the output to a log .txt file
with open("Volt_Bootstrap_logs.txt","w") as f:
    f.write("#Counts number of peaks from 20,000 simulations\n")
    f.write(f"Sample A Voltage Peaks: {peak_counts_sampA}\n")
    f.write(f"Sample B Voltage Peaks: {peak_counts_sampB}\n")
    f.write(f"Sample C Voltage Peaks: {peak_counts_sampC}\n")
    
#----------------------------------------------------------------------------------------------------

#Voltage Mode Confidence Interval
mode_ci_A = np.percentile(bootstrap_sampA_volt_modes, [2.5, 97.5])
mode_ci_B = np.percentile(bootstrap_sampB_volt_modes, [2.5, 97.5])
mode_ci_C = np.percentile(bootstrap_sampC_volt_modes, [2.5, 97.5])
print(mode_ci_A, mode_ci_B, mode_ci_C)
#---------------------------------------------------------------------------------------------------------

##Sample A Voltage total Sum
total_sampA_volt = np.sum((bootstrap_sampA_volt_modes >= mode_ci_A[0]) & 
                          (bootstrap_sampA_volt_modes <= mode_ci_A[1]))
##Divide the Bootstrap Modes into classes
#Sample A Voltage
ci_sampA_volt_low = np.sum((bootstrap_sampA_volt_modes >= mode_ci_A[0])
                & (bootstrap_sampA_volt_modes <= 0.07483))
ci_sampA_volt_medium = np.sum((bootstrap_sampA_volt_modes > 0.07483)
                & (bootstrap_sampA_volt_modes <= 0.08707))
ci_sampA_volt_high = np.sum((bootstrap_sampA_volt_modes > 0.08707)
                & (bootstrap_sampA_volt_modes <= mode_ci_A[1]))
#Percentage number of peaks in low
per_sampA_l = (ci_sampA_volt_low/total_sampA_volt)*100
per_sampA_m = (ci_sampA_volt_medium/total_sampA_volt)*100
per_sampA_h = (ci_sampA_volt_high/total_sampA_volt)*100
perA_all = (per_sampA_l,per_sampA_m, per_sampA_h)
#Robust typical value (median)
median_sampA_l = np.median(bootstrap_sampA_volt_modes[(bootstrap_sampA_volt_modes >= mode_ci_A[0])
                & (bootstrap_sampA_volt_modes <= 0.07483)])
median_sampA_m = np.median(bootstrap_sampA_volt_modes[(bootstrap_sampA_volt_modes > 0.07483)
                & (bootstrap_sampA_volt_modes <= 0.08707)])
median_sampA_h = np.median(bootstrap_sampA_volt_modes[(bootstrap_sampA_volt_modes > 0.08707)
                & (bootstrap_sampA_volt_modes <= mode_ci_A[1])])
medianA_all = (median_sampA_l,median_sampA_m,median_sampA_h)

#Append into the bootstrap log .txt file
with open("Volt_Bootstrap_logs.txt","a") as f:
    f.write("\n--------------------\n")
    f.write("#Sample A Voltage Bootstrap Details\n")
    f.write(f"CI: Lower Bound is {mode_ci_A[0]:.4f}V, Upper Bound is {mode_ci_A[1]:.4f}V")
    f.write("\n")
    f.write("Classification of the Voltage Bootstrap Modes\n")
    f.write(f"Total Voltage Peaks within CI: {total_sampA_volt:.0f}\n")
    f.write(f"Low Class ({mode_ci_A[0]:.4f}V - 0.07483V): {ci_sampA_volt_low:.0f}; Percentage: {per_sampA_l:.4f}%; Median: {median_sampA_l:.4f}V\n")
    f.write(f"Medium Class (> (0.07483)V - 0.08707V): {ci_sampA_volt_medium:.0f}; Percentage: {per_sampA_m:.4f}%; Median: {median_sampA_m:.4f}V\n")
    f.write(f"High Class (0.08707V - {mode_ci_A[1]:.4f}V): {ci_sampA_volt_high:.0f}; Percentage: {per_sampA_h:.4f}%; Median: {median_sampA_h:.4f}V\n")  
#------------------------------------------------------------------------------------------
##Sample B Voltage total Sum
total_sampB_volt = np.sum((bootstrap_sampB_volt_modes >= mode_ci_B[0]) & 
                          (bootstrap_sampB_volt_modes <= mode_ci_B[1]))
##Divide the Bootstrap Modes into classes
#Sample B Voltage
ci_sampB_volt_low = np.sum((bootstrap_sampB_volt_modes >= mode_ci_B[0])
                & (bootstrap_sampB_volt_modes <= 0.10256))
ci_sampB_volt_medium = np.sum((bootstrap_sampB_volt_modes > 0.10256)
                & (bootstrap_sampB_volt_modes <= 0.12316))
ci_sampB_volt_high = np.sum((bootstrap_sampB_volt_modes > 0.12316)
                & (bootstrap_sampB_volt_modes <= mode_ci_B[1]))
#Percentage number of peaks in low
per_sampB_l = (ci_sampB_volt_low/total_sampB_volt)*100
per_sampB_m = (ci_sampB_volt_medium/total_sampB_volt)*100
per_sampB_h = (ci_sampB_volt_high/total_sampB_volt)*100
perB_all = (per_sampB_l,per_sampB_m, per_sampB_h)
#Robust typical value (median)
median_sampB_l = np.median(bootstrap_sampB_volt_modes[(bootstrap_sampB_volt_modes >= mode_ci_B[0])
                & (bootstrap_sampB_volt_modes <= 0.10256)])
median_sampB_m = np.median(bootstrap_sampB_volt_modes[(bootstrap_sampB_volt_modes > 0.10256)
                & (bootstrap_sampB_volt_modes <= 0.12316)])
median_sampB_h = np.median(bootstrap_sampB_volt_modes[(bootstrap_sampB_volt_modes > 0.12316)
                & (bootstrap_sampB_volt_modes <= mode_ci_B[1])])
medianB_all = (median_sampB_l,median_sampB_m,median_sampB_h)

#Append into the bootstrap log .txt file
with open("Volt_Bootstrap_logs.txt","a") as f:
    f.write("\n--------------------\n")
    f.write("#Sample B Voltage Bootstrap Details\n")
    f.write(f"CI: Lower Bound is {mode_ci_B[0]:.4f}V, Upper Bound is {mode_ci_B[1]:.4f}V")
    f.write("\n")
    f.write("Classification of the Voltage Bootstrap Modes\n")
    f.write(f"Total Voltage Peaks within CI: {total_sampB_volt:.0f}\n")
    f.write(f"Low Class ({mode_ci_B[0]:.4f}V - 0.10256V): {ci_sampB_volt_low:.0f}; Percentage: {per_sampB_l:.4f}%; Median: {median_sampB_l:.4f}V\n")
    f.write(f"Medium Class (> (0.10256)V - 0.12316V): {ci_sampB_volt_medium:.0f}; Percentage: {per_sampB_m:.4f}%; Median: {median_sampB_m:.4f}V\n")
    f.write(f"High Class (0.12316V - {mode_ci_B[1]:.4f}V): {ci_sampB_volt_high:.0f}; Percentage: {per_sampB_h:.4f}%; Median: {median_sampB_h:.4f}V\n")  
#------------------------------------------------------------------------------------------

##Sample C Voltage total Sum
total_sampC_volt = np.sum((bootstrap_sampC_volt_modes >= mode_ci_C[0]) & 
                          (bootstrap_sampC_volt_modes <= mode_ci_C[1]))
##Divide the Bootstrap Modes into classes
#Sample C Voltage
ci_sampC_volt_low = np.sum((bootstrap_sampC_volt_modes >= mode_ci_C[0])
                & (bootstrap_sampC_volt_modes <= 0.12543))
ci_sampC_volt_medium = np.sum((bootstrap_sampC_volt_modes > 0.12543)
                & (bootstrap_sampC_volt_modes <= 0.15294))
ci_sampC_volt_high = np.sum((bootstrap_sampC_volt_modes > 0.15294)
                & (bootstrap_sampC_volt_modes <= mode_ci_C[1]))
#Percentage number of peaks in low
per_sampC_l = (ci_sampC_volt_low/total_sampC_volt)*100
per_sampC_m = (ci_sampC_volt_medium/total_sampC_volt)*100
per_sampC_h = (ci_sampC_volt_high/total_sampC_volt)*100
perC_all = (per_sampC_l,per_sampC_m, per_sampC_h)

#Robust typical value (median)
median_sampC_l = np.median(bootstrap_sampC_volt_modes[(bootstrap_sampC_volt_modes >= mode_ci_C[0])
                & (bootstrap_sampC_volt_modes <= 0.12543)])
median_sampC_m = np.median(bootstrap_sampC_volt_modes[(bootstrap_sampC_volt_modes > 0.12543)
                & (bootstrap_sampC_volt_modes <= 0.15294)])
median_sampC_h = np.median(bootstrap_sampC_volt_modes[(bootstrap_sampC_volt_modes > 0.15294)
                & (bootstrap_sampC_volt_modes <= mode_ci_C[1])])
medianC_all = (median_sampC_l,median_sampC_m,median_sampC_h)
#Append into the bootstrap log .txt file
with open("Volt_Bootstrap_logs.txt","a") as f:
    f.write("\n--------------------\n")
    f.write("#Sample C Voltage Bootstrap Details\n")
    f.write(f"CI: Lower Bound is {mode_ci_C[0]:.4f}V, Upper Bound is {mode_ci_C[1]:.4f}V")
    f.write("\n")
    f.write("Classification of the Voltage Bootstrap Modes\n")
    f.write(f"Total Voltage Peaks within CI: {total_sampC_volt:.0f}\n")
    f.write(f"Low Class ({mode_ci_C[0]:.4f}V - 0.12543V): {ci_sampC_volt_low:.0f}; Percentage: {per_sampC_l:.4f}%; Median: {median_sampC_l:.4f}V\n")
    f.write(f"Medium Class (> (0.12543)V - 0.15294V): {ci_sampC_volt_medium:.0f}; Percentage: {per_sampC_m:.4f}%; Median: {median_sampC_m:.4f}V\n")
    f.write(f"High Class (0.15294V - {mode_ci_C[1]:.4f}V): {ci_sampC_volt_high:.0f}; Percentage: {per_sampC_h:.4f}%; Median: {median_sampC_h:.4f}V\n")  
#------------------------------------------------------------------------------------------
##Bootstrap voltage comparison infor for BarChart
#List of infor for each sample
sampA_counts= [per_sampA_l,per_sampA_m,per_sampA_h]
sampB_counts= [per_sampB_l,per_sampB_m,per_sampB_h]
sampC_counts= [per_sampC_l,per_sampC_m,per_sampC_h]

#Labels and bar positions
labels = ['Low','Medium','High']
width_diff = 0.25
x_positions= np.arange(len(labels))

#-------------------------------------------------------------------------------------------------
#Mode CI Bootstrap Plots
mode_fig, mode_axs = plt.subplots(2,2, figsize=(12,8), constrained_layout = True)

#Create over-reaching tiltle for all plots
mode_fig.suptitle(r"\textbf{Bootstrap Mode Confidence Interval for Samples A-C Voltage}", fontsize = 14, weight = 'bold')

#For sample A Voltage
mode_axs[0,0].hist(bootstrap_sampA_volt_modes, bins='auto',alpha = 0.5)
mode_axs[0,0].axvline(mode_ci_A[0],color='red',linestyle='--',label=f'Lower CI: {mode_ci_A[0]:.3f}')
mode_axs[0,0].axvline(mode_ci_A[1],color='green',linestyle='--',label=f'Upper CI: {mode_ci_A[1]:.3f}')
mode_axs[0,0].set_title('(a) Sample A Voltage')
mode_axs[0,0].grid(alpha=0.4)
mode_axs[0,0].legend()
mode_axs[0,0].set_xlabel(r"Voltage $(V)$")
mode_axs[0,0].set_ylabel('Frequency')

#For sample B Voltage
mode_axs[0,1].hist(bootstrap_sampB_volt_modes, bins='auto',alpha = 0.5)
mode_axs[0,1].axvline(mode_ci_B[0],color='red',linestyle='--',label=f'Lower CI: {mode_ci_B[0]:.3f}')
mode_axs[0,1].axvline(mode_ci_B[1],color='green',linestyle='--',label=f'Upper CI: {mode_ci_B[1]:.3f}')
mode_axs[0,1].set_title('(b) Sample B Voltage')
mode_axs[0,1].grid(alpha=0.4)
mode_axs[0,1].legend()
mode_axs[0,1].set_xlabel(r"Voltage $(V)$")
mode_axs[0,1].set_ylabel('Frequency')

#For sample C Voltage
mode_axs[1,0].hist(bootstrap_sampC_volt_modes, bins='auto',alpha = 0.5)
mode_axs[1,0].axvline(mode_ci_C[0],color='red',linestyle='--',label=f'Lower CI: {mode_ci_C[0]:.3f}')
mode_axs[1,0].axvline(mode_ci_C[1],color='green',linestyle='--',label=f'Upper CI: {mode_ci_C[1]:.3f}')
mode_axs[1,0].set_title('(c) Sample C Voltage')
mode_axs[1,0].grid(alpha=0.4)
mode_axs[1,0].legend()
mode_axs[1,0].set_xlabel(r"Voltage $(V)$")
mode_axs[1,0].set_ylabel('Frequency')

#Bar chart for comparison of classifications
mode_axs[1,1].bar(x_positions-width_diff,sampA_counts,width_diff,label='Sample A', color='orange')
mode_axs[1,1].bar(x_positions,sampB_counts,width_diff,label='Sample B',color='red')
mode_axs[1,1].bar(x_positions+width_diff,sampC_counts,width_diff,label='Sample C',color='green')
#Barchart properties
mode_axs[1,1].set_ylim(0,80)
mode_axs[1,1].set_xticks(x_positions)
mode_axs[1,1].set_xticklabels(labels)
mode_axs[1,1].set_title('(d) Voltage Classification Across Samples')
mode_axs[1,1].set_ylabel('Percentage')
mode_axs[1,1].grid(alpha=0.2)
mode_axs[1,1].legend()
#Adding the median text on each bar
for i, (perc,median_value) in enumerate(zip(perA_all,medianA_all)):
    mode_axs[1,1].text(x_positions[i]-width_diff,perc+0.3,fr"${median_value:.4f}\,\mathrm{{V^*}}$",ha='center',va='bottom',fontsize=7)
for i, (perc,median_value) in enumerate(zip(perB_all,medianB_all)):
    mode_axs[1,1].text(x_positions[i],perc+0.3,fr"${median_value:.4f}\,\mathrm{{V^*}}$",ha='center',va='bottom',fontsize=7)
for i, (perc,median_value) in enumerate(zip(perC_all,medianC_all)):
    mode_axs[1,1].text(x_positions[i]+width_diff,perc+0.3,fr"${median_value:.4f}\,\mathrm{{V^*}}$",ha='center',va='bottom',fontsize=7)
plt.savefig("BootstrapMode.png", dpi = 300, bbox_inches = 'tight')
plt.show()

#-------------------------------------------------------------------------------------------------------

##Bootstrap Voltage Median 
n_interations = 20000
start = time.time()
#List of bootstrap median voltage
bootstrap_sampA_volt_median = []
bootstrap_sampB_volt_median = []
bootstrap_sampC_volt_median = []

#A DEF statement to run for all samples, returns the median for each
def bootstrap_median(data,n_iterations):
    return [np.median(np.random.choice(data,size=len(data),replace=True)) for _ in range(n_iterations)]

#Bootstrap for Sample A median
bootstrap_sampA_volt_median = bootstrap_median(sampleA_volt,n_interations)
#Bootstrap for Sample B median
bootstrap_sampB_volt_median = bootstrap_median(sampleB_volt,n_interations)
#Bootstrap for Sample C median
bootstrap_sampC_volt_median = bootstrap_median(sampleC_volt,n_interations)
     
#The total time for boostrapmedian
print(f'Total time is {time.time()-start}')
#-----------------------------------------------------------------------------------------------

#Saving the Median Bootstrap Voltage results
#First convert to numpy for fast computations
bootstrap_sampA_volt_median = np.array(bootstrap_sampA_volt_median)
bootstrap_sampB_volt_median = np.array(bootstrap_sampB_volt_median)
bootstrap_sampC_volt_median = np.array(bootstrap_sampC_volt_median)

#Save to npy and csv file
np.savez("Voltage Bootstrap Median.npz",Sample_A=bootstrap_sampA_volt_median,
         Sample_B=bootstrap_sampB_volt_median,
         Sample_C=bootstrap_sampC_volt_median)
#Padded with NaN to be of the same enght to avoid error
boot_median_volt = pd.DataFrame({"Sample A Bootstrap Median":pd.Series(bootstrap_sampA_volt_median),
                               "Sample B Bootstrap Median":pd.Series(bootstrap_sampB_volt_median),
                               "Sample C Bootstrap Median":pd.Series(bootstrap_sampC_volt_median)})
boot_median_volt.to_csv("Voltage Bootstrap Median.csv", index=False)
#-------------------------------------------------------------------------------------------------

#Count number of medians returned out of 20,000 simulations
median_counts_sampA = len(bootstrap_sampA_volt_median)
median_counts_sampB = len(bootstrap_sampB_volt_median)
median_counts_sampC = len(bootstrap_sampC_volt_median)
print(median_counts_sampA, median_counts_sampB, median_counts_sampC)

#----------------------------------------------------------------------------------------------------

#Voltage Median Confidence Interval
median_ci_A = np.percentile(bootstrap_sampA_volt_median, [2.5, 97.5])
median_ci_B = np.percentile(bootstrap_sampB_volt_median, [2.5, 97.5])
median_ci_C = np.percentile(bootstrap_sampC_volt_median, [2.5, 97.5])
print(median_ci_A, median_ci_B, median_ci_C)
#---------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
##Median CI Bootstrap Plots
median_fig, median_axs = plt.subplots(2,2, figsize=(12,8), constrained_layout = True)

#Create over-reaching tiltle for all plots
median_fig.suptitle(r"\textbf{Bootstrap Median Confidence Interval for Samples A-C Voltage}", fontsize = 14, weight = 'bold')

#Switch off the unsed subplot
median_axs[1,1].axis("off")
#For sample A Voltage
median_axs[0,0].hist(bootstrap_sampA_volt_median, bins='auto',alpha = 0.5)
median_axs[0,0].axvline(median_ci_A[0],color='red',linestyle='--',label=f'Lower CI: {median_ci_A[0]:.3f}')
median_axs[0,0].axvline(median_ci_A[1],color='green',linestyle='--',label=f'Upper CI: {median_ci_A[1]:.3f}')
median_axs[0,0].set_title('(a) Sample A Voltage')
median_axs[0,0].grid(alpha=0.4)
median_axs[0,0].legend()
median_axs[0,0].set_xlabel(r"Voltage $(V)$")
median_axs[0,0].set_ylabel('Frequency')

#For sample B Voltage
median_axs[0,1].hist(bootstrap_sampB_volt_median, bins='auto',alpha = 0.5)
median_axs[0,1].axvline(median_ci_B[0],color='red',linestyle='--',label=f'Lower CI: {median_ci_B[0]:.3f}')
median_axs[0,1].axvline(median_ci_B[1],color='green',linestyle='--',label=f'Upper CI: {median_ci_B[1]:.3f}')
median_axs[0,1].set_title('(b) Sample B Voltage')
median_axs[0,1].grid(alpha=0.4)
median_axs[0,1].legend(loc=left_edge)
median_axs[0,1].set_xlabel(r"Voltage $(V)$")
median_axs[0,1].set_ylabel('Frequency')

#For sample C Voltage
median_axs[1,0].hist(bootstrap_sampC_volt_median, bins='auto',alpha = 0.5)
median_axs[1,0].axvline(median_ci_C[0],color='red',linestyle='--',label=f'Lower CI: {median_ci_C[0]:.3f}')
median_axs[1,0].axvline(median_ci_C[1],color='green',linestyle='--',label=f'Upper CI: {median_ci_C[1]:.3f}')
median_axs[1,0].set_title('(c) Sample C Voltage')
median_axs[1,0].grid(alpha=0.4)
median_axs[1,0].legend()
median_axs[1,0].set_xlabel(r"Voltage $(V)$")
median_axs[1,0].set_ylabel('Frequency')
plt.savefig("BootstrapMedian.png", dpi = 300, bbox_inches = 'tight')
plt.show()


