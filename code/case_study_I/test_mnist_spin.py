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
#import pyNN.nest as p
import spynnaker.pyNN as p
from time import gmtime, strftime

# In[57]:

def plot_spikes(spikes, title):
    if spikes is not None:
        plt.figure(figsize=(15, 5))
        plt.plot([i[1] for i in spikes], [i[0] for i in spikes], ".")
        plt.xlabel('Time/ms')
        plt.ylabel('spikes')
        plt.title(title)

    else:
        print "No spikes received"
    plt.show()

def get_weight_file(folder, digit, cluster, run_num):
    files = []
    files += [each for each in os.listdir(folder) if each.startswith('weight_%d_%d_'%(digit,cluster))]
    weight_file = '%s/weight_%d_%d_%d.npy'%(folder,digit,cluster,(len(files)-1)*run_num)
    return weight_file

def allToall2ConnectList(weights, delays):
    num_in = weights.shape[0]
    num_out = weights.shape[1]
    connect = list()
    for i in range (num_in):
        for j in range (num_out):
            connect.append((i,j,weights[i,j],delays))
    return connect    
# In[58]:

input_size = 28#14
MIN_rate = 1.0

dur_test = 1000 #1000 #ms
silence = 200 #ms

num_train = 30
num_test = 100 
num_cluster = 10
num_digit = 10
test_offset = int(sys.argv[1])
random.seed(test_offset)

num_output = num_cluster*num_digit
cell_params_lif = {'cm': 0.25,
                   'i_offset': 0.0,
                   'tau_m': 20.0,
                   'tau_refrac': 3.0, # 1ms less 2.0
                   'tau_syn_E': 1.0,
                   'tau_syn_I': 1.0,
                   'v_reset': -70.0,
                   'v_rest': -65.0,
                   'v_thresh': -50.0
                   }


# In[59]:
weights_all = 'weights_spikeArray.npy'
if os.path.exists(weights_all):
    trained_weights=np.load(weights_all)    
    print 'found weights file', weights_all
else:
    print 'not found weights file', weights_all
    directory = 'cluster_weights_spikearray'
    
    for i in range(num_digit):
        for j in range(num_cluster):
            weight_file = get_weight_file(directory, i, j, num_train)
            if i==0 and j==0:
                trained_weights = np.load(weight_file)
            else:
                trained_weights = np.append(trained_weights,np.load(weight_file),axis=1)
    np.save(weights_all, trained_weights)
    
  
weight_sum = np.max(trained_weights,axis=0)
weight_max = np.max(weight_sum)
for i in range(num_cluster*num_digit):
    trained_weights[:,i] = trained_weights[:,i]*weight_max/weight_sum[i]*2.0

negtive_weights=np.copy(trained_weights)
index_nz = np.where(negtive_weights > 0.2)
index_z = np.where(negtive_weights <= 0.2)
negtive_weights[index_nz] = 0.0
negtive_weights[index_z] = -0.5



test_x,test_y = pois.get_test_data()
SUM_rate = 2000.
'''
# In[61]:
test_x, test_y = pois.get_test_data()
test_y = np.array(test_y)
test_y = test_y.astype(int)
test_x = np.array(test_x)

Max_rate = 2000.
test_x = test_x.astype(float)
for i in range(len(test_x)):
    test_x[i] = test_x[i]/sum(test_x[i])*Max_rate
'''
# In[62]:

p.setup(timestep=1.0, min_delay=1.0, max_delay=3.0)


'''
# In[63]:
ee_connector = p.OneToOneConnector(weights=3.0)
pop_input = p.Population(input_size*input_size, p.IF_curr_exp, cell_params_lif)
ImagePoission = list()
for i in range(num_test):
    pop = p.Population(input_size*input_size, p.SpikeSourcePoisson,
                                              {'rate' : MIN_rate,#test_x[i],
                                               'start' : (i)*(dur_test+silence),
                                               'duration' : dur_test})
    for j in range(input_size*input_size):
        temp_popv = p.PopulationView(pop, np.array([j]))
        temp_popv.set('rate', test_x[i+test_offset][j])
    ImagePoission.append(pop)


# In[64]:

for i in range(num_test):
    p.Projection(ImagePoission[i], pop_input, ee_connector, target='excitatory')
'''
spike_source_data = pois.mnist_poisson_gen(test_x[test_offset:test_offset+num_test], input_size, input_size, SUM_rate, dur_test, silence)
pop_poisson = p.Population(input_size*input_size, p.SpikeSourceArray,
                                                  {'spike_times' : spike_source_data})

ee_connector = p.OneToOneConnector(weights=3.0)
pop_input = p.Population(input_size*input_size, p.IF_curr_exp, cell_params_lif)
p.Projection(pop_poisson, pop_input, ee_connector, target='excitatory')

# In[65]:

pop_output = p.Population(num_output, p.IF_curr_exp, cell_params_lif)
conn_list_exci = allToall2ConnectList(trained_weights, 1.0)
conn_list_inhi = allToall2ConnectList(negtive_weights, 1.0)
p.Projection(pop_input, pop_output, p.FromListConnector(conn_list_exci), target='excitatory')
p.Projection(pop_input, pop_output, p.FromListConnector(conn_list_inhi), target='inhibitory')
#p.Projection(pop_input, pop_output, p.AllToAllConnector(weights = trained_weights), target='excitatory')
#p.Projection(pop_input, pop_output, p.AllToAllConnector(weights = negtive_weights), target='inhibitory')
print test_offset, strftime("%Y-%m-%d %H:%M:%S", gmtime())


# In[66]:


for i in range(num_output):
    conn_list = list()
    for j in range(num_output):
        if np.ceil(i/num_digit) != np.ceil(j/num_digit):
        #if i!= j:
            conn_list.append((i, j, -1.1, 1.0))
    p.Projection(pop_output, pop_output, p.FromListConnector(conn_list), target='inhibitory')


# In[67]:

pop_output.record()
p.run(num_test*(dur_test+silence))
spikes = pop_output.getSpikes(compatible_output=True)
#plot_spikes(spikes, "output")
p.end()


# In[68]:

spike_count = list()
for i in range(num_output):
    index_i = np.where(spikes[:,0] == i)
    spike_train = spikes[index_i, 1]
    temp = np.histogram(spike_train, bins=range(0, (dur_test+silence)*num_test+1,dur_test+silence))[0]
    spike_count.append(temp)

spike_group = list()
for i in range(num_digit):
    for j in range(num_cluster):
        if j == 0:
            temp = spike_count[i*num_cluster]
        else:
            temp = temp + spike_count[i*num_cluster+j]
    spike_group.append(temp)
# In[69]:

if test_offset==0:
    predict_label = -1*np.ones(10000)
    predict_label2 = -1*np.ones(10000)
else:
    predict_label = np.load('predict_label_spin.npy')
    predict_label2 = np.load('predict_label2_spin.npy')
spike_count = np.array(spike_count)
spike_group = np.array(spike_group)
for i in range(num_test):
    label = np.ceil(np.argmax(spike_count[:,i])/num_digit)
    predict_label[i+test_offset] = label
    
    label = np.argmax(spike_group[:,i])
    predict_label2[i+test_offset] = label
    
#print 'spike_count: ', spike_count, '\npredict_label: ' ,predict_label2[test_offset:test_offset+num_test].astype(int), '\ncorrect_label: ', test_y[test_offset:test_offset+num_test]
np.save('predict_label_spin',predict_label)
np.save('predict_label2_spin',predict_label2)
