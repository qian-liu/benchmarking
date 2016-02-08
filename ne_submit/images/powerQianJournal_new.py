# -*- coding: utf-8 -*-
"""
Created on Mon Jan 26 17:28:27 2015

@author: Vagoir
"""

import numpy as np
import pylab as plt


f3=plt.figure()
ax=f3.add_subplot(111)
lbl=["0","500","1000","1500","2000"]

expresults=[0.27853560652343878,
 0.28559265227641639,
 0.29673561810297999,
 0.30354484649472302,
 0.3080469781324936]

nanojoules = [0.0,735.2634511,392.4909385,274.5132012,213.6658101]

b1=ax.bar([x-0.1 for x in range(0,len(expresults))],expresults,width=0.2,
    color='k',label="Experimental",align="center")

ax.set_ylabel("Power (w)", size=20)
ax.set_xlabel("Input firing rate (Hz)", size=20)

ax2 = ax.twinx()

b2=ax2.bar(
    [x+0.1 for x in range(0,len(expresults))],
    nanojoules,width=0.2,color='b',label="Experimental",align="center")

ax2.set_ylabel("Energy (nJ/SE)", size=20, color='b')
ax2.tick_params(axis='y', labelsize=15)
for tl in ax2.get_yticklabels():
    tl.set_color('b')

ax.set_ylim((0.15,0.32))
ax.set_xlim((-0.5,len((lbl))-0.5))
ax.set_xticks(range(len(lbl)))
ax.set_xticklabels(lbl)
ax.tick_params(axis='y', labelsize=15)
ax.tick_params(axis='x', labelsize=15)



plt.show()
