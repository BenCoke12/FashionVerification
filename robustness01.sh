#!/bin/bash

divisor=1000

now=$(date +"%T")
echo "Current time : $now"

for i in {20..100..10}
do
    #echo "scale=3 ; $i / $divisor" | bc 
    epsilon=$(bc -l <<< "scale=3; $i / $divisor")
    echo "$epsilon"
done

for i in {20..100..10}
do
    now=$(date +"%T")
    echo "Current time : $now"
    epsilon=$(bc -l <<< "scale=3; $i / $divisor")
    timeout 3600 vehicle compileAndVerify --specification vclspecs/fashionRobustness.vcl --network classifier:onnxnetworks/fashion1l.onnx --parameter epsilon:0$epsilon --dataset trainingImages:idxdata/0-49Images.idx --dataset trainingLabels:idxdata/0-49Labels.idx --verifier Marabou --verifierLocation ../Marabou/build/Marabou > onelayer49log0$epsilon.txt && echo "Completed onelayer49-0$epsilon" || echo "timeout onelayer49-0$epsilon"
    now=$(date +"%T")
    echo "Current time : $now"
    timeout 3600 vehicle compileAndVerify --specification vclspecs/fashionRobustness.vcl --network classifier:onnxnetworks/fashion2l.onnx --parameter epsilon:0$epsilon --dataset trainingImages:idxdata/0-49Images.idx --dataset trainingLabels:idxdata/0-49Labels.idx --verifier Marabou --verifierLocation ../Marabou/build/Marabou > twolayer49log0$epsilon.txt && echo "Completed twolayer49-0$epsilon" || echo "timeout twolayer49-0$epsilon"
done

for i in {20..100..10}
do
    now=$(date +"%T")
    echo "Current time : $now"
    epsilon=$(bc -l <<< "scale=3; $i / $divisor")
    timeout 3600 vehicle compileAndVerify --specification vclspecs/fashionRobustness.vcl --network classifier:onnxnetworks/fashion1l.onnx --parameter epsilon:0$epsilon --dataset trainingImages:idxdata/50-99Images.idx --dataset trainingLabels:idxdata/50-99Labels.idx --verifier Marabou --verifierLocation ../Marabou/build/Marabou > onelayer99log0$epsilon.txt && echo "Completed onelayer99-0$epsilon" || echo "timeout onelayer99-0$epsilon"
    now=$(date +"%T")
    echo "Current time : $now"
    timeout 3600 vehicle compileAndVerify --specification vclspecs/fashionRobustness.vcl --network classifier:onnxnetworks/fashion2l.onnx --parameter epsilon:0$epsilon --dataset trainingImages:idxdata/50-99Images.idx --dataset trainingLabels:idxdata/50-99Labels.idx --verifier Marabou --verifierLocation ../Marabou/build/Marabou > twolayer99log0$epsilon.txt && echo "Completed twolayer99-0$epsilon" || echo "timeout twolayer99-0$epsilon"
done