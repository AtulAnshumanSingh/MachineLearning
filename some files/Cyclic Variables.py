# -*- coding: utf-8 -*-
"""
Created on Mon May 20 14:28:52 2019

@author: u346442
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

X = np.array([1, 2, 3, 4, 5, 6, 7])

y0 = np.cos(2*np.pi*(X)/7)
y1 = np.sin(2*np.pi*(X)/7)


fig, ax = plt.subplots()

ax.scatter(y0,y1)

for i, txt in enumerate(X):
    ax.annotate(txt, (y0[i], y1[i]))