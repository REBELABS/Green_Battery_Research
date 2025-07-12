# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 23:09:42 2025

@author: Dr. REBEL-ious
"""
from PIL import Image
import matplotlib.pyplot as plt

#Load individual images
img1 = Image.open('evol_plots.png')
img2 = Image.open('pilot_plot.png')
img3 = Image.open('histogram_plots.png')

#Create sub-plots
fig_A, axs = plt.subplots(2,2, figsize =(12,8), constrained_layout = True)

#Create over-reaching tiltle for all plots
fig_A.suptitle(r"\textbf{Diffusion KDE Analysis for Sample A Voltage}", fontsize = 14, weight = 'bold')

# Display images on subplots
axs[0,0].imshow(img1)
axs[0,0].axis('off')
axs[0,0].set_title("(a) The Optimal diffusion KDE and its evolution stages", fontsize = 12)

axs[0,1].imshow(img2)
axs[0,1].axis('off')
axs[0,1].set_title(r"(b) The diffusion KDE $(T^{m})$ and its pilot estimate", fontsize = 12)

axs[1,0].imshow(img3)
axs[1,0].axis('off')
axs[1,0].set_title("(c) Comparison of the KDEs' performance on the Voltage histogram", fontsize = 12)

# Turn off unused subplot (bottom-right)
axs[1,1].axis('off')

#Show plots
plt.tight_layout()
plt.savefig("SampleAKDE", dpi = 300, bbox_inches = 'tight')
plt.show()

#SampleB
#Load individual images
img1 = Image.open('evol_plotsB.png')
img2 = Image.open('pilot_plotB.png')
img3 = Image.open('histogram_plotsB.png')

#Create sub-plots
fig_A, axs = plt.subplots(2,2, figsize =(12,8), constrained_layout = True)

#Create over-reaching tiltle for all plots
fig_A.suptitle(r"\textbf{Diffusion KDE Analysis for Sample B Voltage}", fontsize = 14, weight = 'bold')

# Display images on subplots
axs[0,0].imshow(img1)
axs[0,0].axis('off')
axs[0,0].set_title("(a) The Optimal diffusion KDE and its evolution stages", fontsize = 12)

axs[0,1].imshow(img2)
axs[0,1].axis('off')
axs[0,1].set_title(r"(b) The diffusion KDE $(T^{*})$ and its pilot estimate", fontsize = 12)

axs[1,0].imshow(img3)
axs[1,0].axis('off')
axs[1,0].set_title("(c) Comparison of the KDEs' performance on the Voltage histogram", fontsize = 12)

# Turn off unused subplot (bottom-right)
axs[1,1].axis('off')

#Show plots
plt.tight_layout()
plt.savefig("SampleBKDE", dpi = 300, bbox_inches = 'tight')
plt.show()

#SampleC
#Load individual images
img1 = Image.open('evol_plotsC.png')
img2 = Image.open('pilot_plotC.png')
img3 = Image.open('histogram_plotsC.png')

#Create sub-plots
fig_A, axs = plt.subplots(2,2, figsize =(12,8), constrained_layout = True)

#Create over-reaching tiltle for all plots
fig_A.suptitle(r"\textbf{Diffusion KDE Analysis for Sample C Voltage}", fontsize = 14, weight = 'bold')

# Display images on subplots
axs[0,0].imshow(img1)
axs[0,0].axis('off')
axs[0,0].set_title("(a) The Optimal diffusion KDE and its evolution stages", fontsize = 12)

axs[0,1].imshow(img2)
axs[0,1].axis('off')
axs[0,1].set_title(r"(b) The diffusion KDE $(T^{m})$ and its pilot estimate", fontsize = 12)

axs[1,0].imshow(img3)
axs[1,0].axis('off')
axs[1,0].set_title("(c) Comparison of the KDEs' performance on the Voltage histogram", fontsize = 12)

# Turn off unused subplot (bottom-right)
axs[1,1].axis('off')

#Show plots
plt.tight_layout()
plt.savefig("SampleCKDE", dpi = 300, bbox_inches = 'tight')
plt.show()

#Currents
#Sample B Current
#Load individual images
img1 = Image.open('evol_plotsBcurrent.png')
img2 = Image.open('pilot_plotBcurrent.png')
img3 = Image.open('histogram_plotsBcurrent.png')

#Create sub-plots
fig_A, axs = plt.subplots(2,2, figsize =(12,8), constrained_layout = True)

#Create over-reaching tiltle for all plots
fig_A.suptitle(r"\textbf{Diffusion KDE Analysis for Sample B Current}", fontsize = 14, weight = 'bold')

# Display images on subplots
axs[0,0].imshow(img1)
axs[0,0].axis('off')
axs[0,0].set_title("(a) The Optimal diffusion KDE and its evolution stages", fontsize = 12)

axs[0,1].imshow(img2)
axs[0,1].axis('off')
axs[0,1].set_title(r"(b) The diffusion KDE $(T^{m})$ and its pilot estimate", fontsize = 12)

axs[1,0].imshow(img3)
axs[1,0].axis('off')
axs[1,0].set_title("(c) Comparison of the KDEs' performance on the Current histogram", fontsize = 12)

# Turn off unused subplot (bottom-right)
axs[1,1].axis('off')

#Show plots
plt.tight_layout()
plt.savefig("SampleBCKDE", dpi = 300, bbox_inches = 'tight')
plt.show()

#SampleC Current
#Load individual images
img1 = Image.open('evol_plotsCcurrent.png')
img2 = Image.open('pilot_plotCcurrent.png')
img3 = Image.open('histogram_plotsCcurrent.png')

#Create sub-plots
fig_A, axs = plt.subplots(2,2, figsize =(12,8), constrained_layout = True)

#Create over-reaching tiltle for all plots
fig_A.suptitle(r"\textbf{Diffusion KDE Analysis for Sample C Current}", fontsize = 14, weight = 'bold')

# Display images on subplots
axs[0,0].imshow(img1)
axs[0,0].axis('off')
axs[0,0].set_title("(a) The Optimal diffusion KDE and its evolution stages", fontsize = 12)

axs[0,1].imshow(img2)
axs[0,1].axis('off')
axs[0,1].set_title(r"(b) The diffusion KDE $(T^{m})$ and its pilot estimate", fontsize = 12)

axs[1,0].imshow(img3)
axs[1,0].axis('off')
axs[1,0].set_title("(c) Comparison of the KDEs' performance on the Current histogram", fontsize = 12)

# Turn off unused subplot (bottom-right)
axs[1,1].axis('off')

#Show plots
plt.tight_layout()
plt.savefig("SampleCCKDE", dpi = 300, bbox_inches = 'tight')
plt.show()


#------------------------------------------------------------------------------
#Modes for all sample's voltage
#Load individual images
img1 = Image.open('ModesA.png')
img2 = Image.open('ModesB.png')
img3 = Image.open('ModesC.png')
img4 = Image.open('compositeKDEs_volt.png')

#Create sub-plots
fig_B, axsb = plt.subplots(2,2, figsize =(12,8), constrained_layout = True)

#Create over-reaching tiltle for all plots
fig_B.suptitle(r"\textbf{Diffusion KDE Modes Analysis for Sample A-C Voltage}", fontsize = 14, weight = 'bold')

# Display images on subplots
axsb[0,0].imshow(img1)
axsb[0,0].axis('off')
axsb[0,0].set_title("(a) Identified voltage modes and bounding limit for sample A", fontsize = 12)

axsb[0,1].imshow(img2)
axsb[0,1].axis('off')
axsb[0,1].set_title(r"(b) Identified voltage modes and bounding limit for sample B", fontsize = 12)

axsb[1,0].imshow(img3)
axsb[1,0].axis('off')
axsb[1,0].set_title("(c) Identified voltage modes and bounding limit for sample C", fontsize = 12)

axsb[1,1].imshow(img4)
axsb[1,1].axis('off')
axsb[1,1].set_title("(d) Voltage Modes comparison for sample A-C", fontsize = 12)

#Show plots
plt.tight_layout()
plt.savefig("volt_modes_comparison.png", dpi = 300, bbox_inches = 'tight')
plt.show()

#------------------------------------------------------------------------------
#Modes for all sample's voltage
#Load individual images
img1 = Image.open('ModesA.png')
img2 = Image.open('ModesB.png')
img3 = Image.open('ModesC.png')
img4 = Image.open('compositeKDEs_volt.png')

#Create sub-plots
fig_B, axsb = plt.subplots(2,2, figsize =(12,8), constrained_layout = True)

#Create over-reaching tiltle for all plots
fig_B.suptitle(r"\textbf{Diffusion KDE Modes Analysis for Sample A-C Voltage}", fontsize = 14, weight = 'bold')

# Display images on subplots
axsb[0,0].imshow(img1)
axsb[0,0].axis('off')
axsb[0,0].set_title("(a) Identified voltage modes and bounding limit for sample A", fontsize = 12)

axsb[0,1].imshow(img2)
axsb[0,1].axis('off')
axsb[0,1].set_title(r"(b) Identified voltage modes and bounding limit for sample B", fontsize = 12)

axsb[1,0].imshow(img3)
axsb[1,0].axis('off')
axsb[1,0].set_title("(c) Identified voltage modes and bounding limit for sample C", fontsize = 12)

axsb[1,1].imshow(img4)
axsb[1,1].axis('off')
axsb[1,1].set_title("(d) Voltage Modes comparison for sample A-C", fontsize = 12)

#Show plots
plt.tight_layout()
plt.savefig("volt_modes_comparison.png", dpi = 300, bbox_inches = 'tight')
plt.show()