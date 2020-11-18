#!/bin/bash

# create directory only if it doesn't exist
mkdir -p files
mkdir -p files/add_two_matrix
FILE="../files/add_two_matrix/$1"
if [ -f "$FILE" ]; then
    rm -f $FILE
fi
N_v=500
numBlock_v=4
numThreadsPerBlock_v=1
for i in `seq 0 50 3000`; do
    sudo $(which nvcc) ../add_two_matrix.cu --run -D N=$N_v, -D numBlock=$numBlock_v, -D numThreadsPerBlock=$numThreadsPerBlock_v >> $FILE
    numThreadsPerBlock_v=$((numThreadsPerBlock_v + 1))

    #if [ $i -eq "1000" ]; then
        #numThreadsPerBlock_v=1
        #numBlock_v=2
    #fi
    #if [ $i -eq "2000" ]; then
        #numThreadsPerBlock_v=1
        #numBlock_v=3
    #fi

    if [ $numThreadsPerBlock_v -eq "15" ]; then
        rm -f a.out
	    exit 2
    fi
done

# delete file created
rm -f a.out
