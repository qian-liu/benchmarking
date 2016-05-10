# -*- coding: utf-8 -*-
"""
Created on Mon Jan 26 17:28:27 2015

@author: Vagoir
"""

import numpy as np
import pylab as plt

sim_time=30 #seconds

f3=plt.figure()
ax=f3.add_subplot(111)
lbl=["0","500","1000","1500","2000"]

expresults=np.asarray([0.27853560652343878,
 0.28559265227641639,
 0.29673561810297999,
 0.30354484649472302,
 0.3080469781324936])
expresults*=sim_time

# new results with the baseline and idle
nanojoules = [0.0,#[0.0,735.2634511,392.4909385,274.5132012,213.6658101]
16.26101913,
19.91261234,
18.36953859,
16.42784159,
]

meanFiringRates_perlayer = [
[0.0, 0.0, 0.0],
[500.0/784.0, 1.6579158,	3.6862276,	4.72521], #500 spikes per image
[1000.0/784.0,3.074434,	6.4390332,	7.25981], #1000 spikes per image
[1500.0/784.0,4.3479016,	8.6078574,	8.81907], #1500
[2000.0/784.0,5.5108708,	10.3956398,	9.90549], #2000
]
meanFiringRates_perlayer=np.asarray(meanFiringRates_perlayer)
num_of_SE = []
for se in meanFiringRates_perlayer:
	num_of_SE.append( se[0]*784*500+se[1]*500*500+se[2]*500*10 )


b1=ax.bar([x-0.1 for x in range(0,len(expresults))],expresults,width=0.2,
    color='k',label="Experimental",align="center")

#ax.set_ylabel("Power (w)", size=20)
ax.set_ylabel("Energy (J)", size=20)

ax.set_xlabel("Input firing rate (Hz)", size=20)

ax2 = ax.twinx()

b2=ax2.bar(
    [x+0.1 for x in range(0,len(expresults))],
    np.asarray(num_of_SE)/1000,width=0.2,color='b',label="Experimental",align="center")

ax2.set_ylabel("# of kSE", size=20, color='b')
ax2.tick_params(axis='y', labelsize=15)
for tl in ax2.get_yticklabels():
    tl.set_color('b')

ax.set_ylim((0.0,10))
ax2.set_ylim((0,(max(np.asarray(num_of_SE)/1000))+100))
ax.set_xlim((-0.5,len((lbl))-0.5))
ax.set_xticks(range(len(lbl)))
ax.set_xticklabels(lbl)
ax.tick_params(axis='y', labelsize=15)
ax.tick_params(axis='x', labelsize=15)
ax2.yaxis.labelpad=10


plt.show()
