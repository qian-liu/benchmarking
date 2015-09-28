'''
Get the test results:
linear neurons
LIF neurons with nest
LIF neurons with spinnaker
Typical Usage:
number of clusters per digit: 10
training simulator: nest
test simulator: nest
duration per test: 1000 (ms)
firing rate for the whole image: 5000 (Hz)
python test_analysis.py $num_cluster $sim_train $sim_test $dur $sum_rate
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
from time import gmtime, strftime

def plot_spikes(spikes, title):
    if spikes is not None:
        plt.figure(figsize=(15, 5))
        plt.plot([i[1] for i in spikes], [i[0] for i in spikes], ".")
        plt.xlabel('Time/ms')
        plt.ylabel('spikes')
        plt.title(title)

    else:
        print "No spikes received"

# get trained weights for single output 
def get_weight_file(folder, digit, cluster):
    weight_file = '%s/weight_%d_%d.npy'%(folder,digit,cluster)
    return weight_file



num_cluster = int(sys.argv[1])
sim_train = sys.argv[2]
sim_test = sys.argv[3]
dur_test = int(sys.argv[4]) 
SUM_rate = float(sys.argv[5]) 
num_digit = 10 # MNIST 10 digits
input_size = 28
silence = 200 #ms
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

# Find the trained weights
if sim_train not in ['spin', 'nest']:
    sys.exit()
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
    
# normalisation and negative weights
weight_sum = np.max(trained_weights,axis=0)
weight_max = np.max(weight_sum)
for i in range(num_cluster*num_digit):
    if weight_sum[i] > 0.:
        trained_weights[:,i] = trained_weights[:,i]*weight_max/weight_sum[i]*2.0
index_z = np.where(trained_weights <= 0.2)
trained_weights[index_z] = -1.0

# Plot the weights
'''
for j in range(num_cluster):
    to_plot = np.transpose(np.reshape(trained_weights[:,j*num_digit:(j+1)*num_digit],(28,28*num_digit),1))
    if j==0:
        plot_list = to_plot
    else:
        plot_list = np.append(plot_list,to_plot,axis=1)

plt.figure(figsize=(15,15))
img = plt.imshow(plot_list ,cmap=cm.gray_r)
#plt.colorbar(img, fraction=0.046, pad=0.04)
plt.axis('off')
plt.show()
'''

# Test Analysis
if sim_train not in ['spin', 'nest', 'linear']:
    sys.exit()

test_x, test_y = pois.get_test_data()
result_file_max = 'predict_%d_%s_%s_max.npy'%(num_cluster, sim_train, sim_test)
result_file_sum = 'predict_%d_%s_%s_sum.npy'%(num_cluster, sim_train, sim_test)
respond_file = 'respond_%d_%s_%s_rate%d_dur%d.npy'%(num_cluster, sim_train, sim_test, SUM_rate, dur_test)
if sim_test in ['spin', 'nest']:
    if os.path.exists(result_file_max):
        predict_label=np.load(result_file_max)
        print 'prediction file found:', result_file_max
        accuracy = np.sum(predict_label == test_y)/100.
        print 'The accuracy over MNIST tested with %s (MAX) is %.2f%%'%(sim_test, accuracy)
    else:
        print 'prediction file not found:', result_file_max
    
    # sum instead of sum
    #if os.path.exists(result_file_sum):
    #    predict_label=np.load(result_file_sum)
    #    print 'prediction file found:', result_file_sum
    #    accuracy = np.sum(predict_label == test_y)/100.
    #    print 'The accuracy over MNIST tested with %s (SUM) is %.2f%%'%(sim_test, accuracy)
    #else:
    #    print 'prediction file not found:', result_file_sum
        
    if os.path.exists(respond_file):
        latency=np.load(respond_file)
        print 'respond file found:', respond_file
        respond_time = np.average(latency)
        print 'The average latency is %.2fms'%(respond_time)
    else:
        print 'respond file not found:', respond_file

elif sim_test == 'linear':
    test_y = np.array(test_y)
    test_y = test_y.astype(int)
    test_x = np.array(test_x)

    Max_rate = 2000.
    test_x = test_x.astype(float)
    for i in range(len(test_x)):
        test_x[i] = test_x[i]/sum(test_x[i])*Max_rate
        
    score = np.zeros((len(test_x), num_digit*num_cluster))
    #score_sum = np.zeros((len(test_x), num_digit))
    for i in range(len(test_x)):
        for j in range(num_digit):
            for k in range(num_cluster):
                ind = j*num_cluster+k
                score[i][ind] = np.sum(test_x[i] * trained_weights[:,ind])
                #score_sum[i][j] += score[i][ind]
    result = np.ceil(np.argmax(score, axis = 1)/num_cluster)
    accuracy = np.sum(result == test_y)/100.
    print 'The accuracy over MNIST tested with linear neurons (MAX) is %.2f%%'%(accuracy)
    #result_sum = np.argmax(score_sum, axis = 1)
    #accuracy = np.sum(result_sum == test_y)/100.
    #print 'The accuracy over MNIST tested with linear neurons (SUM) is %.2f%%'%(accuracy)
    #print score_sum, result, result_sum, test_y

