#!/bin/bash

divisor=1000

now=$(date +"%T")
echo "Current time : $now"

for i in {0..9..1}
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
#for i in {0..99..1}
#do
#    start=`date +%s`
#    timeout 3600 vehicle verify --specification vclspecs/fashionRobustness.vcl --network classifier:onnxnetworks/fashion1l.onnx --parameter epsilon:0.01 --dataset image:idxdata/individuals/Image$i.idx --dataset label:idxdata/individuals/Label$i.idx --verifier Marabou --verifierLocation ../Marabou/build/Marabou > logs/fashion1l/onelayer0.01-$i.txt && echo "Completed onelayer0.01-$i" || echo "timeout onelayer0.01$i"
#    end=`date +%s`
#    echo "Runtime on 1l-0.01-image:$i: $((end-start))" > logs/times/log1l-0.01-$i.txt
#done
#epsilon 0.05
for i in {12..99..1}
do
    start=`date +%s`
    vehicle verify --specification vclspecs/fashionRobustness.vcl --network classifier:onnxnetworks/fashion1l.onnx --parameter epsilon:0.05 --dataset image:idxdata/individuals/Image$i.idx --dataset label:idxdata/individuals/Label$i.idx --verifier Marabou --verifierLocation ../Marabou/build/Marabou > logs/fashion1l/onelayer0.05-$i.txt
    end=`date +%s`
    echo "Runtime on 1l-0.05-image:$i: $((end-start))" > logs/times/log1l-0.05-$i.txt
done
#epsilon 0.1
for i in {0..99..1}
do
    start=`date +%s`
    vehicle verify --specification vclspecs/fashionRobustness.vcl --network classifier:onnxnetworks/fashion1l.onnx --parameter epsilon:0.1 --dataset image:idxdata/individuals/Image$i.idx --dataset label:idxdata/individuals/Label$i.idx --verifier Marabou --verifierLocation ../Marabou/build/Marabou > logs/fashion1l/onelayer0.1-$i.txt
    end=`date +%s`
    echo "Runtime on 1l-0.1-image:$i: $((end-start))" > logs/times/log1l-0.1-$i.txt
done
#epsilon 0.5
for i in {0..99..1}
do
    start=`date +%s`
    vehicle verify --specification vclspecs/fashionRobustness.vcl --network classifier:onnxnetworks/fashion1l.onnx --parameter epsilon:0.5 --dataset image:idxdata/individuals/Image$i.idx --dataset label:idxdata/individuals/Label$i.idx --verifier Marabou --verifierLocation ../Marabou/build/Marabou > logs/fashion1l/onelayer0.5-$i.txt 
    end=`date +%s`
    echo "Runtime on 1l-0.5-image:$i: $((end-start))" > logs/times/log1l-0.1-$i.txt
done