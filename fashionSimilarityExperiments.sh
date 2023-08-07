vehicle --version
#similarity
for i in {0..499..1}
do
    echo "Image: $i"
    start=`date +%s`
    vehicle verify --specification vclspecs/fashionSimilarityOnData.vcl --network classifier:onnxnetworks/fashion1l32n.onnx --dataset images:idxdata/individuals/Image$i.idx --verifier Marabou --verifierLocation ../Marabou/build/Marabou
done