#!/bin/bash

 
# training on nest
for i in `seq 0 9`;
do
    for j in `seq 0 9`;
    do
        python train_spikeArray.py $i $j
    done
done
# run with nest first
for i in `seq 0 10 9999`;
do
    python test_mnist.py $i
done
python test_analysis_spikeArray.py

# training on SpiNNaker
for i in `seq 0 9`;
do
    for j in `seq 0 9`;
    do
        python train_all_spin.py $i $j
    done
done

# run with spinnaker
for i in `seq 0 100 9999`;
do
    python test_mnist_spin.py $i
done
python test_analysis_spin.py



