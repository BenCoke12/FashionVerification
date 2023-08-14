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
for i in {0..499..1}
do
    now=$(date +"%T")
    echo "Current time : $now"
    start=`date +%s`
    unbuffer vehicle verify --specification vclspecs/fashionRobustness.vcl --network classifier:onnxnetworks/pgdTrainedB.onnx --parameter epsilon:0.01 --dataset imageDataset:idxdata/individuals/Image$i.idx --dataset labelDataset:idxdata/individuals/Label$i.idx --verifier Marabou --verifierLocation ../Marabou/build/Marabou > logs/pgdTrainedB/onelayer32n0.01-$i.txt 
    end=`date +%s`
    echo "Runtime on 1l32n-0.01-image:$i: $((end-start))" > logs/pgdTrainedB/times/log1l32n-0.01-$i.txt
done

#epsilon 0.05
for i in {0..499..1}
do
    now=$(date +"%T")
    echo "Current time : $now"
    start=`date +%s`
    unbuffer vehicle verify --specification vclspecs/fashionRobustness.vcl --network classifier:onnxnetworks/pgdTrainedB.onnx --parameter epsilon:0.05 --dataset imageDataset:idxdata/individuals/Image$i.idx --dataset labelDataset:idxdata/individuals/Label$i.idx --verifier Marabou --verifierLocation ../Marabou/build/Marabou > logs/pgdTrainedB/onelayer32n0.05-$i.txt 
    end=`date +%s`
    echo "Runtime on 1l32n-0.05-image:$i: $((end-start))" > logs/pgdTrainedB/times/log1l32n-0.05-$i.txt
done

#epsilon 0.1
for i in {0..499..1}
do
    now=$(date +"%T")
    echo "Current time : $now"
    start=`date +%s`
    unbuffer vehicle verify --specification vclspecs/fashionRobustness.vcl --network classifier:onnxnetworks/pgdTrainedB.onnx --parameter epsilon:0.1 --dataset imageDataset:idxdata/individuals/Image$i.idx --dataset labelDataset:idxdata/individuals/Label$i.idx --verifier Marabou --verifierLocation ../Marabou/build/Marabou > logs/pgdTrainedB/onelayer32n0.1-$i.txt
    end=`date +%s`
    echo "Runtime on 1l32n-0.1-image:$i: $((end-start))" > logs/pgdTrainedB/times/log1l32n-0.1-$i.txt
done

#epsilon 0.5
for i in {0..499..1}
do
    now=$(date +"%T")
    echo "Current time : $now"
    start=`date +%s`
    unbuffer vehicle verify --specification vclspecs/fashionRobustness.vcl --network classifier:onnxnetworks/pgdTrainedB.onnx --parameter epsilon:0.5 --dataset imageDataset:idxdata/individuals/Image$i.idx --dataset labelDataset:idxdata/individuals/Label$i.idx --verifier Marabou --verifierLocation ../Marabou/build/Marabou > logs/pgdTrainedB/onelayer32n0.5-$i.txt 
    end=`date +%s`
    echo "Runtime on 1l32n-0.5-image:$i: $((end-start))" > logs/pgdTrainedB/times/log1l32n-0.5-$i.txt
done