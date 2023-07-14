#first 100 images for robustness
#epsilon 0.01
for i in {0..99..1}
do
    now=$(date +"%T")
    echo "Current time : $now"
    start=`date +%s`
    vehicle verify --specification vclspecs/fashionRobustness.vcl --network classifier:onnxnetworks/fashion1l32n.onnx --parameter epsilon:0.01 --dataset image:idxdata/individuals/Image$i.idx --dataset label:idxdata/individuals/Label$i.idx --verifier Marabou --verifierLocation ../Marabou/build/Marabou > logs/fashion1l32n/onelayer32n0.01-$i.txt 
    end=`date +%s`
    echo "Runtime on 1l32n-0.01-image:$i: $((end-start))" > logs/times/log1l32n-0.01-$i.txt
done

#epsilon 0.05
for i in {0..99..1}
do
    now=$(date +"%T")
    echo "Current time : $now"
    start=`date +%s`
    vehicle verify --specification vclspecs/fashionRobustness.vcl --network classifier:onnxnetworks/fashion1l32n.onnx --parameter epsilon:0.05 --dataset image:idxdata/individuals/Image$i.idx --dataset label:idxdata/individuals/Label$i.idx --verifier Marabou --verifierLocation ../Marabou/build/Marabou > logs/fashion1l32n/onelayer32n0.05-$i.txt 
    end=`date +%s`
    echo "Runtime on 1l32n-0.05-image:$i: $((end-start))" > logs/times/log1l32n-0.05-$i.txt
done

#epsilon 0.1
for i in {0..99..1}
do
    now=$(date +"%T")
    echo "Current time : $now"
    start=`date +%s`
    vehicle verify --specification vclspecs/fashionRobustness.vcl --network classifier:onnxnetworks/fashion1l32n.onnx --parameter epsilon:0.1 --dataset image:idxdata/individuals/Image$i.idx --dataset label:idxdata/individuals/Label$i.idx --verifier Marabou --verifierLocation ../Marabou/build/Marabou > logs/fashion1l32n/onelayer32n0.1-$i.txt
    end=`date +%s`
    echo "Runtime on 1l32n-0.1-image:$i: $((end-start))" > logs/times/log1l32n-0.1-$i.txt
done

#epsilon 0.5
for i in {0..99..1}
do
    now=$(date +"%T")
    echo "Current time : $now"
    start=`date +%s`
    vehicle verify --specification vclspecs/fashionRobustness.vcl --network classifier:onnxnetworks/fashion1l32n.onnx --parameter epsilon:0.5 --dataset image:idxdata/individuals/Image$i.idx --dataset label:idxdata/individuals/Label$i.idx --verifier Marabou --verifierLocation ../Marabou/build/Marabou > logs/fashion1l32n/onelayer32n0.5-$i.txt 
    end=`date +%s`
    echo "Runtime on 1l32n-0.5-image:$i: $((end-start))" > logs/times/log1l32n-0.5-$i.txt
done


#similarity
for i in {0..499..1}
do
    now=$(date +"%T")
    echo "Current time : $now"
    start=`date +%s`
    vehicle verify --specification vclspecs/fashionSimilarityOnData.vcl --network classifier:onnxnetworks/fashion1l32n.onnx --dataset images:idxdata/individuals/Image$i.idx --verifier Marabou --verifierLocation ../Marabou/build/Marabou #> logs/similarity/onelayer32n-$i.txt 
    end=`date +%s`
    #echo "Runtime on 1l32n-0.01-image:$i: $((end-start))" > logs/similarity/times/log1l32n-0.01-$i.txt
done