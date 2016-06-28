'''
Typical Usage:
number of cluster: 10
training simulator: nest
testing simulator: nest
number of testing images: 1000
offset of testing images: 0
python test_mnist.py 10 nest nest 1000 0
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys
import scipy.io as sio
import scipy.cluster.vq as spvq
import scipy.spatial.distance as spdt
import os
import poisson_tools as pois
import random
from time import gmtime, strftime
import time
# In[57]:

def plot_spikes(spikes, title):
    if spikes is not None:
        fig, ax = plt.subplots()
        #plt.figure(figsize=(15, 5))
        ax.plot([i[1] for i in spikes], [i[0] for i in spikes], ".")
        plt.xlabel('Time/ms')
        plt.ylabel('spikes')
        ax.set_xticks(range(0,12000,1200)) 
        ax.xaxis.grid(True)
        ax.set_yticks(range(0,100,10)) 
        ax.yaxis.grid(True)
        plt.title(title)

    else:
        print "No spikes received"
    plt.show()

def allToall2ConnectList(weights, delays):
    num_in = weights.shape[0]
    num_out = weights.shape[1]
    connect = list()
    for i in range (num_in):
        for j in range (num_out):
            connect.append((i,j,weights[i,j],delays))
    return connect    

# get trained weights for single output 
def get_weight_file(folder, digit, cluster):
    weight_file = '%s/weight_%d_%d.npy'%(folder,digit,cluster)
    return weight_file

num_cluster = int(sys.argv[1])
sim_train = sys.argv[2]
sim_test = sys.argv[3]
num_test = int(sys.argv[4])
test_offset = int(sys.argv[5])

input_size = 28
#dur_test = 1000 #1000 #ms
dur_test = int(sys.argv[6])
silence = 200 #ms
num_digit = 10
#SUM_rate = 2000.
SUM_rate = float(sys.argv[7])

random.seed(test_offset)

num_output = num_cluster*num_digit
cell_params_lif = {'cm': 0.25,
                   'i_offset': 0.0,
                   'tau_m': 20.0,
                   'tau_refrac': 2.0,
                   'tau_syn_E': 1.0,
                   'tau_syn_I': 1.0,
                   'v_reset': -70.0,
                   'v_rest': -65.0,
                   'v_thresh': -50.0
                   }


# Load trained weights
weights_all = 'weights_%d_%s.npy'%(num_cluster, sim_train)
if os.path.exists(weights_all):
    trained_weights=np.load(weights_all)
    print 'found weights file', weights_all
else:
    print 'not found weights file', weights_all
    directory = 'cluster_weights_%d_%s'%(num_cluster, sim_train)
    if not os.path.exists(directory):
        print 'The SNN has not been trained yet.'
        sys.exit()
    for i in range(num_digit):
        for j in range(num_cluster):
            weight_file = get_weight_file(directory, i, j)
            if i==0 and j==0:
                trained_weights = np.load(weight_file)
            else:
                trained_weights = np.append(trained_weights,np.load(weight_file),axis=1)
    np.save(weights_all, trained_weights)
    print 'Created weights file', weights_all
    
'''
weight_sum = np.max(trained_weights,axis=0)
weight_max = np.max(weight_sum)

for i in range(num_cluster*num_digit):
    if weight_sum[i] > 2.:
        trained_weights[:,i] = trained_weights[:,i]*weight_max/weight_sum[i]#*2.0
'''

negtive_weights=np.copy(trained_weights)
neg_thr = 0.1
index_nz = np.where(negtive_weights > neg_thr)
index_z = np.where(negtive_weights <= neg_thr)
trained_weights[index_z] = 0.0
negtive_weights[index_nz] = 0.0
negtive_weights[index_z] = -1.5 #-0.5
#trained_weights = trained_weights * 10.
#negtive_weights = negtive_weights * 10.
test_x,test_y = pois.get_test_data()
if sim_test == 'nest':
    import pyNN.nest as p
elif sim_test == 'spin':
    import spynnaker.pyNN as p
else:
    sys.exit()
p.setup(timestep=1.0, min_delay=1.0, max_delay=3.0)
spike_source_data = pois.mnist_poisson_gen(test_x[test_offset:test_offset+num_test], input_size, input_size, SUM_rate, dur_test, silence)

if sim_test == 'nest':
    pop_poisson = p.Population(input_size*input_size, p.SpikeSourceArray, {'spike_times' : []})
    for j in range(input_size*input_size):
        pop_poisson[j].spike_times = spike_source_data[j]
    ee_connector = p.OneToOneConnector(weights=3.0)
    pop_input = p.Population(input_size*input_size, p.IF_curr_exp, cell_params_lif)
    p.Projection(pop_poisson, pop_input, ee_connector, target='excitatory')
    pop_output = p.Population(num_output, p.IF_curr_exp, cell_params_lif)
    p.Projection(pop_input, pop_output, p.AllToAllConnector(weights = trained_weights), target='excitatory')
    p.Projection(pop_input, pop_output, p.AllToAllConnector(weights = negtive_weights), target='inhibitory')

    
elif sim_test == 'spin':
    p.set_number_of_neurons_per_core("IF_curr_exp", 127)
    pop_poisson = p.Population(input_size*input_size, p.SpikeSourceArray, {'spike_times' : spike_source_data})
    ee_connector = p.OneToOneConnector(weights=3.0)
    pop_input = p.Population(input_size*input_size, p.IF_curr_exp, cell_params_lif)
    p.Projection(pop_poisson, pop_input, ee_connector, target='excitatory')
    pop_output = p.Population(num_output, p.IF_curr_exp, cell_params_lif)
    conn_list_exci = allToall2ConnectList(trained_weights, 1.0)
    conn_list_inhi = allToall2ConnectList(negtive_weights, 1.0)
    p.Projection(pop_input, pop_output, p.FromListConnector(conn_list_exci), target='excitatory')
    p.Projection(pop_input, pop_output, p.FromListConnector(conn_list_inhi), target='inhibitory')
    


'''
for i in range(num_output):
    conn_list = list()
    for j in range(num_output):
        #if np.ceil(i/num_digit) != np.ceil(j/num_digit):
        if i!= j:
            conn_list.append((i, j, -1.1, 1.0)) #-1.1
    p.Projection(pop_output, pop_output, p.FromListConnector(conn_list), target='inhibitory')
    print 'WTA:, ', i
'''
pop_output.record()
p.run(num_test*(dur_test+silence))
spikes = pop_output.getSpikes(compatible_output=True)

spike_count = list()
for i in range(num_output):
    index_i = np.where(spikes[:,0] == i)
    spike_train = spikes[index_i, 1]
    temp = np.histogram(spike_train, bins=range(0, (dur_test+silence)*num_test+1,dur_test+silence))[0]
    spike_count.append(temp)

result_file_max = 'predict_%d_%s_%s_max.npy'%(num_cluster, sim_train, sim_test)
#result_file_sum = 'predict_%d_%s_%s_sum.npy'%(num_cluster, sim_train, sim_test)
respond_file = 'respond_%d_%s_%s_rate%d_dur%d.npy'%(num_cluster, sim_train, sim_test, SUM_rate, dur_test)
if test_offset==0:
    predict_max = -1*np.ones(10000)
    #predict_sum = -1*np.ones(10000)
    respond_time = 1000*np.ones(10000)
else:
    predict_max = np.load(result_file_max)
    #predict_sum = np.load(result_file_sum)
    respond_time = np.load(respond_file)
spike_count = np.array(spike_count)
#spike_group = np.array(spike_group)
spike_source = np.sort(np.ceil(np.concatenate(spike_source_data))).astype(int)
for i in range(num_test):
    if max(spike_count[:,i]) > 0:
        label = np.ceil(np.argmax(spike_count[:,i])/num_cluster)
        predict_max[i+test_offset] = label
        #print 'MAX: ', label
        #label = np.argmax(spike_group[:,i])
        #predict_sum[i+test_offset] = label
        #print 'SUM: ', label
        if len(spikes) > 0:
            resp_train = np.where(spikes[:,1]>i*(dur_test+silence))[0]
            if len(resp_train) > 0:
                resp_ind = resp_train[0]
                source_ind = np.where(spike_source>i*(dur_test+silence))[0][0]
                latency = spikes[resp_ind, 1] - spike_source[source_ind]
                respond_time[i+test_offset] = latency
np.save(result_file_max,predict_max)
#np.save(result_file_sum,predict_sum)
np.save(respond_file,respond_time)

p.end()
