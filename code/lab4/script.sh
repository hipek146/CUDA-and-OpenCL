#!/bin/bash
string=''
echo -n "" > results.txt
n=2
for (( threads=1; $threads <= 10; threads++ )) ; do
	string=''
	for (( blocks=1; $blocks <= 15; blocks++ )) ; do
		n_threads=$(($n**$threads))
		n_blocks=$(($n**$blocks))
		sum=0
		for (( i=1; $i <= 10; i++ )) ; do
			nvprof --log-file output.txt ./lab4 $1 $n_threads $n_blocks 1 > /dev/null 2>&1
			value=$(grep 'Stride' ./output.txt | awk {'print $6'})
			numb=$(echo ${value: : -2} + $sum)
			sum=$(echo $numb | bc -l)
		done
		scale="scale=4;"
		string+=$(echo $scale $sum / 10 | bc -l)
		string+=$'\t'
	done
	echo $string >> results.txt
done

sum=0
for (( i=1; $i <= 10; i++ )) ; do
	nvprof --log-file output.txt ./lab4 $1 256 100 2 > /dev/null 2>&1
	value=$(grep 'Mismatch' ./output.txt | awk {'print $6'})
	numb=$(echo ${value: : -2} + $sum)
	sum=$(echo $numb | bc -l)
done
scale="scale=4;"
echo $scale $sum / 10 | bc -l
