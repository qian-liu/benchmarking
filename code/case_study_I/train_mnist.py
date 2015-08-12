'''
Typical Usage:
number of clusters per digit = 10
Train digit 0, cluster 0
Simualator: Nest
python train_mnist.py 10 0 0 nest
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
import time

# given the k-means centers, classify data into nearest cluster
def belong_cluster(data, cluster_centers):
    MAX_dist = 1000
    num_cluster = len(cluster_centers)
    distance = MAX_dist*np.ones(num_cluster)
    for i in range(num_cluster):
        distance[i] = spdt.euclidean(data, cluster_centers[i])
    k_index= np.argmin(distance)
    return k_index

# translate saved weights into connection list
def allToall2ConnectList(weights, delays):
    num_in = weights.shape[0]
    num_out = weights.shape[1]
    connect = list()
    for i in range (num_in):
        for j in range (num_out):
            connect.append((i,j,weights[i,j],delays))
    return connect    

def get_scale(num_sample, num_cluster):
    if num_sample == 0:
        return 0
    else:
        return 6000./num_cluster/num_sample
    
# load MNIST training database
train_x,train_y = pois.get_train_data()
label_list = np.array(train_y)
label_list = label_list.astype(int)
num_digit = 10 # MNIST 10 digits
num_cluster = int(sys.argv[1])
digit = int(sys.argv[2])
cluster = int(sys.argv[3])
sim = sys.argv[4]
random.seed(digit*num_digit + cluster)

if sim == 'nest':
    import pyNN.nest as p
elif sim == 'spin':
    import spynnaker.pyNN as p
else:
    sys.exit()

# creat cluster index file if neccessary
fname = 'index_cluster_%d.npy'%(num_cluster)
if not os.path.isfile(fname):
    print 'no', fname, 'file found, begin creating it.'

    index_digit_list = list()
    k_center_list = list()
    for i in range(num_digit):
        index_digit = np.where(label_list==i)[0]
        # random seed deos not work on spvq.kmeans
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

#load clustered index file
index_cluster = np.load(fname)
index_digit = np.where(label_list==digit)[0]
index_train = np.where(index_cluster[index_digit]==cluster)[0]
input_size = 28
#dur_train = 30 * num_cluster #ms
scale = get_scale(len(index_train), num_cluster)
dur_train = 30 * num_cluster * scale#ms
silence = 0 #ms
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
# Training with Spiking Neural Network
# pop_poisson -> pop_input ---STDP--> pop_output <- pop_teach
print digit, cluster, len(index_train)
p.setup(timestep=1.0, min_delay=1.0, max_delay=16.0)
pop_input = p.Population(input_size*input_size, p.IF_curr_exp, cell_params_lif)
pop_output = p.Population(1, p.IF_curr_exp, cell_params_lif)

# generate poissonian Spike Source Array for pop_teach and pop_poisson 
spike_teach = pois.poisson_generator(TEACH_rate, 0, len(index_train)*(dur_train+silence))
spike_source_data = pois.mnist_poisson_gen(train_x[index_digit[index_train]], input_size, input_size, SUM_rate, dur_train, silence)
if sim == 'spin':
    pop_teach = p.Population(1, p.SpikeSourceArray, {'spike_times' : spike_teach})
    pop_poisson = p.Population(input_size*input_size, p.SpikeSourceArray, {'spike_times' : spike_source_data})

elif sim == 'nest':
    pop_teach = p.Population(1, p.SpikeSourceArray, {'spike_times' : spike_teach})
    pop_poisson = p.Population(input_size*input_size, p.SpikeSourceArray,
                                                  {'spike_times' : []})
    for j in range(input_size*input_size):
        pop_poisson[j].spike_times = spike_source_data[j]
ee_connector = p.OneToOneConnector(weights=3.0)
p.Projection(pop_poisson, pop_input, ee_connector, target='excitatory')
p.Projection(pop_teach, pop_output, ee_connector, target='excitatory')
weight_max = 1.3
stdp_model = p.STDPMechanism(
    timing_dependence=p.SpikePairRule(tau_plus=10.0, tau_minus=10.0),
    weight_dependence=p.MultiplicativeWeightDependence(w_min=0.0, w_max=weight_max, A_plus=0.01, A_minus=0.01)
)
proj_stdp = p.Projection(
    pop_input, pop_output, p.AllToAllConnector(weights = 0.0),
    synapse_dynamics=p.SynapseDynamics(slow=stdp_model))

start = time.time()
p.run(len(index_train)*(dur_train+silence))
end = time.time()
b_time = len(index_train)*(dur_train+silence)
sim_str = 'training time:%.4f s, biology time:%d ms\n'%(end-start, b_time)
f=open('log_t5.txt','a')
f.write(sim_str)
f.close()
# save trained weights into a folder
directory = 'cluster_weights_%d_%s'%(num_cluster,sim)
if not os.path.exists(directory):
    os.makedirs(directory)
post = proj_stdp.getWeights(format='array',gather=False)
np.save('%s/weight_%d_%d.npy'%(directory,digit,cluster),post)

p.end()



