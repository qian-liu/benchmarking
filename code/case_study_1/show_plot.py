import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
#spike_count = np.load('groups.npy')
#plt.figsize=(100,100)
#plt.imshow(spike_count,interpolation='none',cmap = cm.Greys_r)
#plt.show()
'''
import matplotlib.pyplot as plt
acc_time = [63.54, 79.11, 84.45, 86.53, 87.40, 88.46, 88.88, 89.33, 89.90, 90.03, 90.30, 90.95, 91.26, 91.21]
time_acc = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000]
acc_time_10 = [77.04, 85.14, 88.29, 88.97, 89.39, 89.89, 90.11, 90.20, 90.52, 90.76, 91.12, 91.43, 91.43, 91.38]
time_acc_10 = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000]
# Accuracy over time
plt.plot(time_acc,acc_time,'bo',markersize=5.0)
plt.plot(time_acc,acc_time,linewidth=1.0)
plt.plot(time_acc_10,acc_time_10,'k^',markersize=5.0)
plt.plot(time_acc_10,acc_time_10,'k', linewidth=1.0)
plt.xlabel('Time per image (ms)')
plt.ylabel('Classification accuracy (%)')
plt.ylim((75,95))
plt.grid(True)
plt.show()

acc_rate = [9.98, 29.14, 54.73, 74.85, 84.06, 88.28, 90.03, 91.27, 91.42, 91.45]
rate_acc = [800, 1000, 1200, 1400, 1600, 1800, 2000, 3000, 4000, 5000]
acc_rate_10 = [16.42, 41.61, 62.31, 73.15, 80.41, 83.57, 85.67, 86.82, 88.04, 88.93, 90.76, 90.70, 90.15, 90.04  ]
rate_acc_10 = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000]
plt.plot(rate_acc,acc_rate,'bo',markersize=5.0)
plt.plot(rate_acc,acc_rate,'b',linewidth=1.0)
plt.plot(rate_acc_10,acc_rate_10,'k^',markersize=5.0)
plt.plot(rate_acc_10,acc_rate_10,'k',linewidth=1.0)
plt.xlabel('Spiking rate per image (Hz)')
plt.ylabel('Classification accuracy (%)')
plt.ylim((60,95))
plt.grid(True)
plt.show()

latency_rate = [ 933.58, 753.88, 491.53, 276.66, 150.07, 86.81, 58.47, 23.97, 17.35, 14.20]#2451
rate_latency = [800, 1000, 1200, 1400, 1600, 1800, 2000, 3000, 4000, 5000]
latency_rate_10 = [533.29, 194.02, 94.14, 59.28, 42.04, 32.87, 27.08, 23.48, 20.61, 18.49, 10.81, 8.45, 7.30, 6.61 ]#791.30, 198.12, 
rate_latency_10 = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000] #100, 200, 
plt.plot(rate_latency,latency_rate,'bo',markersize=5.0)
plt.plot(rate_latency,latency_rate,'b',linewidth=1.0)
plt.plot(rate_latency_10,latency_rate_10,'k^',markersize=5.0)
plt.plot(rate_latency_10,latency_rate_10,'k',linewidth=1.0)
plt.xlabel('Spiking rate per image (Hz)')
plt.ylabel('Latency (ms)')
plt.ylim((0, 200))
plt.grid(True)
plt.show()

latency_time = [ 74.00, 59.94, 58.43, 58.87, 57.40, 57.27, 57.72, 57.40, 58.20, 58.47, 58.75, 58.69, 58.19, 58.65]
time_latency = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000]
latency_time_10 = [ 10.73, 10.74, 10.75, 10.79, 10.74, 10.68, 10.69, 10.73, 10.70, 10.81, 10.74, 10.74, 10.79, 10.70]
time_latency_10 = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000]
plt.plot(time_latency,latency_time,'bo',markersize=5.0)
plt.plot(time_latency,latency_time,'b',linewidth=1.0)
plt.plot(time_latency_10,latency_time_10,'k^',markersize=5.0)
plt.plot(time_latency_10,latency_time_10,'k',linewidth=1.0)
plt.xlabel('Time per image (ms)')
plt.ylabel('Latency (ms)')
plt.ylim((0,80))
plt.grid(True)
plt.show()




'''
num_cluster = [1, 10, 50, 100, 1000]# 1000, 20, 30, 40, 50, 60, 70, 80, 90]
energy_nest = np.array([554.77, 621.74, 1125.12, 1406.01, 3036.88])*21/1000.#, 89.65]
energy_spin = np.array([0.38, 0.38, 0.41, 0.44, 1.50])*12.000#, 89.65]

index = np.arange(5)
bar_width = 0.35
plt.bar(index, energy_nest, bar_width, color='b', label='NEST')
plt.bar(index+bar_width, energy_spin, bar_width, color='k', label='SpiNNaker')
#plt.plot(num_cluster,energy_nest,'bo',markersize=5.0)
#plt.plot(num_cluster,energy_nest,linewidth=1.0)
#plt.plot(num_cluster,energy_spin,'g^',markersize=5.0)
#plt.plot(num_cluster,energy_spin,linewidth=1.0)
plt.xlabel('Number of clusters/subclasses per digit')
plt.ylabel('Energy use (KJ)')
plt.xticks(index + bar_width, ('1', '10', '50', '100', '1000'))
#plt.xscale('log')
#plt.legend()
plt.tight_layout()
plt.gca().yaxis.grid(True)
#plt.grid(True)
plt.show()

