#!/bin/bash

divisor=1000

now=$(date +"%T")
echo "Current time : $now"

for i in {2..9..1}
do
    #echo "scale=3 ; $i / $divisor" | bc 
    epsilon=$(bc -l <<< "scale=3; $i / $divisor")
    start=`date +%s`
    echo "$i"
    #sleep 1
    end=`date +%s`
    #echo "Runtime on $i: $((end-start))" > log$i.txt
done

#epsilon 0.01
for i in {0..1..1}
do
    now=$(date +"%T")
    echo "Current time : $now"
    start=`date +%s`
    unbuffer vehicle verify --specification vclspecs/fashionSuperclassRobustness.vcl --network classifier:onnxnetworks/fashion1l32n.onnx --parameter epsilon:0.01 --dataset trainingImages:idxdata/individuals/Image$i.idx --dataset trainingLabels:idxdata/individuals/Label$i.idx --verifier Marabou --verifierLocation ../Marabou/build/Marabou > logs/superclass/results/32n0.01-$i.txt 
    end=`date +%s`
    echo "Runtime on 1l32n-0.01-image:$i: $((end-start))" > logs/superclass/times/32n-0.01-$i.txt
done
