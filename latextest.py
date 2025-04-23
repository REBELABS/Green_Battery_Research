# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 09:55:51 2025

@author: user
"""
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True

plt.plot([1, 2, 3], [1, 4, 9], label=r"$y = x^2$")
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.title(r"\textbf{Test Plot using LaTeX}")
plt.legend()
plt.grid(True)
plt.show()
