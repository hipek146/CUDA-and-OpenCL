#!/bin/bash

path=$(pwd)
filenames=$(ls "${path}" | grep ".cu")
#echo $filenames
#IFS=' ' read -ra filearray <<< "$filenames"; declare -p filearray;
#echo $array
for file in $filenames
do
	chmod 744 ${file}
	echo ${file}
	execname=$(echo ${file} | cut -d '.' -f 1)
	echo ${execname}
	`nvcc -arch=sm_75 -std=c++11 -o "${execname}.exe" ${file}`
done
execnames=$(ls "${path}" | grep ".exe")
echo $execnames

for e in $execnames
do
	resultname=$(echo ${e} | cut -d '.' -f 1)
	[ -e "${resultname}.txt" ] && rm "${resultname}.txt"; touch "${resultname}.txt" || touch "${resultname}.txt"
	for i in {10..24..2}
	do
		./${e} ${i} "${resultname}.txt"
	done
done