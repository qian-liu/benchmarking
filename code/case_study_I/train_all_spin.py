import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys
import scipy.io as sio
import scipy.cluster.vq as spvq
import scipy.spatial.distance as spdt
import os
import poisson_tools as pois
#import pyNN.nest as p
import spynnaker.pyNN as p
import random

def belong_cluster(data, cluster_centers):
    MAX_dist = 1000
    num_cluster = len(cluster_centers)
    distance = MAX_dist*np.ones(num_cluster)
    for i in range(num_cluster):
        distance[i] = spdt.euclidean(data, cluster_centers[i])
    k_index= np.argmin(distance)
    return k_index

def import_spinnaker():
    import spynnaker.pyNN as p
    return p

def allToall2ConnectList(weights, delays):
    num_in = weights.shape[0]
    num_out = weights.shape[1]
    connect = list()
    for i in range (num_in):
        for j in range (num_out):
            connect.append((i,j,weights[i,j],delays))
    return connect    
    
train_x,train_y = pois.get_train_data()
label_list = np.array(train_y)
label_list = label_list.astype(int)
num_digit = 10
num_cluster = 10
digit = int(sys.argv[1])
cluster = int(sys.argv[2])
random.seed(digit*num_digit + cluster)

# In[16]:
fname = 'index_cluster.npy'
if not os.path.isfile(fname):
    print 'no', fname, 'file found, begin creating it.'

    index_digit_list = list()
    k_center_list = list()
    for i in range(num_digit):
        index_digit = np.where(label_list==i)[0]
        k_center = spvq.kmeans(train_x[index_digit], num_cluster)
        index_digit_list.append(index_digit)
        k_center_list.append(k_center)
        print i
    
    num_train_img = 60000 #len(num_train_img)
    index_cluster = np.zeros(num_train_img)
    digit_list = np.zeros(num_train_img)
    for i in range(num_train_img):
        data = train_x[i]
        cluster_centers = k_center_list[label_list[i]][0]
        index_cluster[i] = belong_cluster(data, cluster_centers)
        
    np.save(fname, index_cluster)


# In[17]:

index_cluster = np.load(fname)

#digit = 0
#cluster = 0
index_digit = np.where(label_list==digit)[0]
index_train = np.where(index_cluster[index_digit]==cluster)[0]
input_size = 28

dur_train = 300 #ms
silence = 0 #ms

num_per_run = 30

TEACH_rate = 50.0
SUM_rate = 2000.0
cell_params_lif = {'cm': 0.25,
                   'i_offset': 0.0,
                   'tau_m': 20.0,
                   'tau_refrac': 3.0,   # 2.0
                   'tau_syn_E': 1.0,
                   'tau_syn_I': 1.0,
                   'v_reset': -70.0,
                   'v_rest': -65.0,
                   'v_thresh': -50.0
                   }

print digit, cluster, len(index_train)
p.setup(timestep=1.0, min_delay=1.0, max_delay=16.0)
pop_input = p.Population(input_size*input_size, p.IF_curr_exp, cell_params_lif)
pop_output = p.Population(1, p.IF_curr_exp, cell_params_lif)
spike_teach = pois.poisson_generator(TEACH_rate, 0, len(index_train)*(dur_train+silence))
pop_teach = p.Population(1, p.SpikeSourceArray, {'spike_times' : spike_teach})
spike_source_data = pois.mnist_poisson_gen(train_x[index_digit[index_train]], input_size, input_size, SUM_rate, dur_train, silence)
pop_poisson = p.Population(input_size*input_size, p.SpikeSourceArray, {'spike_times' : spike_source_data})
ee_connector = p.OneToOneConnector(weights=3.0)
p.Projection(pop_poisson, pop_input, ee_connector, target='excitatory')
p.Projection(pop_teach, pop_output, ee_connector, target='excitatory')
weight_max = 1.3
stdp_model = p.STDPMechanism(
    timing_dependence=p.SpikePairRule(tau_plus=10.0, tau_minus=10.0),
    weight_dependence=p.MultiplicativeWeightDependence(w_min=0.0, w_max=weight_max, A_plus=0.01, A_minus=0.01)
)

directory = 'cluster_weights_spin'
if not os.path.exists(directory):
    os.makedirs(directory)
proj_stdp = p.Projection(
    pop_input, pop_output, p.AllToAllConnector(weights = 0.0),
    synapse_dynamics=p.SynapseDynamics(slow=stdp_model))

p.run(len(index_train)*(dur_train+silence))
post = proj_stdp.getWeights(format='array',gather=False)
np.save('%s/weight_%d_%d.npy'%(directory,digit,cluster),post)
p.end()



