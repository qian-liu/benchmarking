import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys
import scipy.io as sio
import scipy.cluster.vq as spvq
import scipy.spatial.distance as spdt
import os
import poisson_tools as pois
import pyNN.nest as p
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

def get_weight_file(folder, digit, cluster, run_num):
    files = []
    files += [each for each in os.listdir(folder) if each.startswith('weight_%d_%d_'%(digit,cluster))]
    weight_file = '%s/weight_%d_%d_%d.npy'%(folder,digit,cluster,(len(files)-1)*run_num)
    return weight_file
# In[58]:

input_size = 28#14
MIN_rate = 1.0

dur_test = 1000 #ms
silence = 200 #ms

num_test = 10
num_cluster = 10
num_digit = 10

#test_offset = int(sys.argv[1])

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


# In[59]:
weights_all = 'weights_spikeArray.npy'
if os.path.exists(weights_all):
    trained_weights=np.load(weights_all)
    print 'found weights file', weights_all
else:
    print 'not found weights file', weights_all
    #directory = 'cluster_weights_nest'
    directory = 'cluster_weights_spikearray'
    num_run = 30
    for i in range(num_digit):
        for j in range(num_cluster):
            weight_file = get_weight_file(directory, i, j, num_run)
            if i==0 and j==0:
                trained_weights = np.load(weight_file)
            else:
                trained_weights = np.append(trained_weights,np.load(weight_file),axis=1)
    np.save(weights_all, trained_weights)
    

weight_sum = np.max(trained_weights,axis=0)
weight_max = np.max(weight_sum)
for i in range(num_cluster*num_digit):
    trained_weights[:,i] = trained_weights[:,i]*weight_max/weight_sum[i]*2.0
index_z = np.where(trained_weights <= 0.2)
trained_weights[index_z] = -0.5
'''     
negtive_weights=np.copy(trained_weights)
index_nz = negtive_weights.nonzero()
index_z = np.where(negtive_weights == 0)
negtive_weights[index_nz] = 0.0
negtive_weights[index_z] = -0.1*3.0
''' 

for j in range(num_cluster):
    to_plot = np.transpose(np.reshape(trained_weights[:,j*num_digit:(j+1)*num_digit],(28,28*num_digit),1))
    if j==0:
        plot_list = to_plot
    else:
        plot_list = np.append(plot_list,to_plot,axis=1)
    #print plot_list.shape
'''
plt.figure(figsize=(15,15))
img = plt.imshow(plot_list ,cmap=cm.gray_r)
#plt.colorbar(img, fraction=0.046, pad=0.04)
plt.axis('off')
plt.show()
'''
test_x, test_y = pois.get_test_data()
result_file = 'predict_label.npy'
if os.path.exists(result_file):
    predict_label=np.load(result_file)
    print 'predict_label file found:', result_file
    accuracy = np.sum(predict_label == test_y)/100.
    print 'The accuracy over MNIST tested with Spiking Neural Networks (MAX) is %.2f%%'%(accuracy)
    #str_tmp = 'The accuracy over MNIST tested with Spiking Neural Networks is %.2f%%\n'%(accuracy)
    str_tmp = '%.2f\n'%(accuracy)
    
    result_file = 'predict_label2.npy'
    predict_label=np.load(result_file)
    print 'predict_label file found:', result_file
    accuracy = np.sum(predict_label == test_y)/100.
    print 'The accuracy over MNIST tested with Spiking Neural Networks (SUM) is %.2f%%'%(accuracy)
    #str_tmp = 'The accuracy over MNIST tested with Spiking Neural Networks is %.2f%%\n'%(accuracy)
    str_tmp2 = '%.2f\n'%(accuracy)
    
else:
    test_y = np.array(test_y)
    test_y = test_y.astype(int)
    test_x = np.array(test_x)

    Max_rate = 2000.
    test_x = test_x.astype(float)
    for i in range(len(test_x)):
        test_x[i] = test_x[i]/sum(test_x[i])*Max_rate
        
    score = np.zeros((len(test_x), num_digit*num_cluster))
    for i in range(len(test_x)):
        for j in range(num_digit*num_cluster):
            score[i][j] = np.sum(test_x[i] * trained_weights[:,j])
            
    result = np.ceil(np.argmax(score, axis = 1)/num_cluster)
    #label = np.argmax(test_y, axis = 1)
    accuracy = np.sum(result == test_y)/100.
    print 'The accuracy over MNIST tested with linear neurons is %.2f%%'%(accuracy)
    #str_tmp = 'The accuracy over MNIST tested with linear neurons is %.2f%%\n'%(accuracy)
    str_tmp = '%.2f\n'%(accuracy)
    
f_acc = 'accuracy.txt'
f = open(f_acc, "a")
f.write(str_tmp)
f.write(str_tmp2)
f.close()
