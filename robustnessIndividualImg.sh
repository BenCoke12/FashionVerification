#!/bin/bash

divisor=1000

now=$(date +"%T")
echo "Current time : $now"

for i in {0..10..1}
do
    #echo "scale=3 ; $i / $divisor" | bc 
    epsilon=$(bc -l <<< "scale=3; $i / $divisor")
    echo "$i"
done