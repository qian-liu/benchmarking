import matplotlib.pyplot as plt
import numpy as np
def setAxLinesBW(ax):
    """
    Take each Line2D in the axes, ax, and convert the line style to be 
    suitable for black and white viewing.
    """
    MARKERSIZE = 8

    COLORMAP = {
        'b': {'marker': '^', 'dash': (None,None)},
        'g': {'marker': 'v', 'dash': [5,5]},
        'r': {'marker': None, 'dash': [5,3,1,3]},
        'c': {'marker': None, 'dash': [1,3]},
        'm': {'marker': None, 'dash': [5,2,5,2,5,10]},
        'y': {'marker': None, 'dash': [5,3,1,2,1,10]},
        'k': {'marker': 'o', 'dash': (None,None)} #[1,2,1,10]}
        }


    lines_to_adjust = ax.get_lines()
    try:
        lines_to_adjust += ax.get_legend().get_lines()
    except AttributeError:
        pass

    for line in lines_to_adjust:
        origColor = line.get_color()
        line.set_color('black')
        line.set_dashes(COLORMAP[origColor]['dash'])
        line.set_marker(COLORMAP[origColor]['marker'])
        line.set_markersize(MARKERSIZE)

def setFigLinesBW(fig):
    """
    Take each axes in the figure, and for each line in the axes, make the
    line viewable in black and white.
    """
    for ax in fig.get_axes():
        setAxLinesBW(ax)
        
import matplotlib.pyplot as plt
import numpy as np
fontsize = 20
linesize = 3

acc_time = [63.54, 79.11, 84.45, 86.53, 87.40, 88.46, 88.88, 89.33, 89.90, 90.01, 90.30, 90.95, 91.26, 91.21]
time_acc = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000]
acc_time_10 = [77.04, 85.14, 88.29, 88.97, 89.39, 89.89, 90.11, 90.20, 90.52, 90.76, 91.12, 91.43, 91.43, 91.38]
time_acc_10 = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000]
# Accuracy over time

fig = plt.figure(figsize=(7,5))
ax = fig.add_subplot(111)#(321)
ax.plot(time_acc,acc_time, linewidth=linesize, label='Original Weights')
ax.plot(time_acc_10,acc_time_10, linewidth=linesize, label='Scaled-up Weights')
ax.legend(loc='lower right', fontsize=fontsize)
ax.grid('on')
ax.set_xlabel('Time / image (ms)',size=fontsize)
ax.set_ylabel('Classification accuracy (%)',size=fontsize)
ax.set_ylim((75,95))
ax.tick_params( labelsize=fontsize)
setFigLinesBW(fig)
plt.tight_layout()
fig.savefig('../../images/acc_dur.pdf')

acc_rate = [9.98, 29.14, 54.73, 74.85, 84.06, 88.28, 89.94, 91.27, 91.42, 91.45]
rate_acc = [800, 1000, 1200, 1400, 1600, 1800, 2000, 3000, 4000, 5000]
acc_rate_10 = [16.42, 41.61, 62.31, 73.15, 80.41, 83.57, 85.67, 86.82, 88.04, 88.93, 90.76, 90.70, 90.15, 90.04  ]
rate_acc_10 = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000]

fig = plt.figure(figsize=(7,5))
ax = fig.add_subplot(111)#(322)
ax.plot(rate_acc,acc_rate, linewidth=linesize)
ax.plot(rate_acc_10,acc_rate_10, linewidth=linesize)
ax.grid('on')
ax.set_xlabel('Input firing rate (Hz)',size=fontsize)
ax.set_ylabel('Classification accuracy (%)',size=fontsize)
ax.set_ylim((60,95))
ax.tick_params( labelsize=fontsize)
setFigLinesBW(fig)
plt.tight_layout()
fig.savefig('../../images/acc_rate.pdf')


latency_rate = [ 933.58, 753.88, 491.53, 276.66, 150.07, 86.81, 58.47, 23.97, 17.35, 14.20]#2451
rate_latency = [800, 1000, 1200, 1400, 1600, 1800, 2000, 3000, 4000, 5000]
latency_rate_10 = [533.29, 194.02, 94.14, 59.28, 42.04, 32.87, 27.08, 23.48, 20.61, 18.49, 10.81, 8.45, 7.30, 6.61 ]#791.30, 198.12, 
rate_latency_10 = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000] #100, 200, 

fig = plt.figure(figsize=(7,5))
ax = fig.add_subplot(111)#(324)
ax.plot(rate_latency,latency_rate, linewidth=linesize)
ax.plot(rate_latency_10,latency_rate_10, linewidth=linesize)
ax.grid('on')
ax.set_xlabel('Input firing rate (Hz)',size=fontsize)
ax.set_ylabel('Latency (ms)',size=fontsize)
ax.set_ylim((0, 200))
ax.tick_params( labelsize=fontsize)
setFigLinesBW(fig)
plt.tight_layout()
fig.savefig('../../images/lat_rate.pdf')

latency_time = [ 74.00, 59.94, 58.43, 58.87, 57.40, 57.27, 57.72, 57.40, 58.20, 58.47, 58.75, 58.69, 58.19, 58.65]
time_latency = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000]
latency_time_10 = [ 10.73, 10.74, 10.75, 10.79, 10.74, 10.68, 10.69, 10.73, 10.70, 10.81, 10.74, 10.74, 10.79, 10.70]
time_latency_10 = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000]

fig = plt.figure(figsize=(7,5))
ax = fig.add_subplot(111)#(323)
ax.plot(time_latency,latency_time, linewidth=linesize)
ax.plot(time_latency_10,latency_time_10, linewidth=linesize)
ax.grid('on')
ax.set_xlabel('Time / image (ms)',size=fontsize)
ax.set_ylabel('Latency (ms)',size=fontsize)
ax.set_ylim((0, 80))
ax.tick_params( labelsize=fontsize)
setFigLinesBW(fig)
plt.tight_layout()
fig.savefig('../../images/lat_dur.pdf')

num_event = np.load('time_events.npy')
num_event/=600000.
num_event+=400
num_event_10 = np.load('time_events_10.npy')
num_event_10/=600000.
num_event_10+=400
time = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000]

fig = plt.figure(figsize=(7,5))
ax = fig.add_subplot(111)#(325)
ax.plot(time,num_event, linewidth=linesize)
ax.plot(time,num_event_10, linewidth=linesize)
ax.grid('on')
ax.set_xlabel('Time / image (ms)',size=fontsize)
ax.set_ylabel('Sopbs (kHz)',size=fontsize)
ax.set_ylim((400,407))
ax.tick_params( labelsize=fontsize)
setFigLinesBW(fig)
plt.tight_layout()
fig.savefig('../../images/time_event.pdf')

num_event = np.load('rate_events.npy')
num_event/=600000.
num_event_10 = np.load('rate_events_10.npy')
num_event_10/=600000.
rates = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000]
num_event+=np.array(rates)*0.2
num_event_10+=np.array(rates)*0.2

fig = plt.figure(figsize=(7,5))
ax = fig.add_subplot(111)#(326)
ax.plot(rates,num_event, linewidth=linesize)
ax.plot(rates,num_event_10, linewidth=linesize)
ax.grid('on')
ax.set_xlabel('Input firing rate (Hz)',size=fontsize)
ax.set_ylabel('Sopbs (kHz)',size=fontsize)
# ax.set_ylim((0,1200))
ax.tick_params( labelsize=fontsize)
setFigLinesBW(fig)
plt.tight_layout()
fig.savefig('../../images/rate_event.pdf')


'''
num_cluster = [1, 10, 50, 100, 1000]# 1000, 20, 30, 40, 50, 60, 70, 80, 90]
energy_nest = np.array([445.14*20, 502.77*20, 766.96*20, 1131.67*19, 12249.84*17])/1000.#, 89.65]
energy_spin = np.array([0.38, 0.38, 0.41, 0.44, 1.50])*12.000#, 89.65]
plt.clf()
index = np.arange(5)
bar_width = 0.35
plt.yscale('log')
plt.bar(index, energy_nest, bar_width, color='b', label='NEST on a PC', log='true')
plt.bar(index+bar_width, energy_spin, bar_width, color='k', label='SpiNNaker',log='true')
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
plt.legend(loc='upper left', shadow=True)

plt.show()
'''
