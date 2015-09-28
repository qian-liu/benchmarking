#!/bin/bash

num_digit=10
sim_train=nest
num_cluster=50

# SOURCE virtual environment for NEST or SpiNNaker
#source ~/apt/spin_1408/virtualenv/bin/activate
source ~/env/nest_pynn0.7/bin/activate

sum_rate=5000
python test_analysis.py $num_cluster $sim_train linear 0 $sum_rate >> log.txt

sim_test=nest
num_test=100
dur=1000

echo '--Start testing with '$sim_test', dur_test: '$dur', num_cluster: '$num_cluster', sum_rate: '$sum_rate', '$num_test' per test' >> log.txt
date >> log.txt
        
for i in `seq 0 $num_test 9999`;
do
    python test_mnist.py  $num_cluster $sim_train $sim_test $num_test $i $dur $sum_rate
    echo '-Done: test' $i >> log.txt
    date >> log.txt
done
        
echo '--Finish testing with' $sim_test >> log.txt
date >> log.txt
python test_analysis.py $num_cluster $sim_train $sim_test $dur $sum_rate>> log.txt
