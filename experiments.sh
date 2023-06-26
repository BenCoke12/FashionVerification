#!/bin/bash

divisor=1000
for i in {0..20..2}
do
    #echo "scale=3 ; $i / $divisor" | bc 
    epsilon=$(bc -l <<< "scale=3; $i / $divisor")
    echo "$epsilon"
done

#for i in {0..20..2}
#do
#    now=$(date +"%T")
#    echo "Current time : $now"
#    epsilon=$(bc -l <<< "scale=3; $i / $divisor")
#    vehicle compileAndVerify --specification vclspecs/fashionRobustness.vcl --network classifier:onnxnetworks/fashion2l.onnx --parameter epsilon:0$epsilon --dataset trainingImages:idxdata/50-99Images.idx --dataset trainingLabels:idxdata/50-99Labels.idx --verifier Marabou --verifierLocation ../Marabou/build/Marabou > twolayer99log0$epsilon.txt
#done

now=$(date +"%T")
echo "Current time : $now"

for i in {20..100..10}
do
    #echo "scale=3 ; $i / $divisor" | bc 
    epsilon=$(bc -l <<< "scale=3; $i / $divisor")
    echo "$epsilon"
done
