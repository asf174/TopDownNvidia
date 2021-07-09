#!/bin/bash

for i in $(seq 1 1 10); do
    (time topdown.py -l3 -nd  -f ./dwt2d "192.bmp -d 192x192 -f -5 -l 3") >> time_topdown_192.txt 2>&1
    (time ./dwt2d 192.bmp -d 192x192 -f -5 -l 3) >> time_native_192.txt 2>&1
done
 topdown.py -l3 -nd -os scan_file_192.txt -o results_topdown_192.txt -f ./dwt2d "192.bmp -d 192x192 -f -5 -l 3"

for i in $(seq 1 1 10); do
    (time topdown.py -l3 -nd  -f ./dwt2d "rgb.bmp -d 1024x1024 -f -5 -l 3") >> time_topdown.txt 2>&1
    (time ./dwt2d rgb.bmp -d 1024x1024 -f -5 -l 3) >> time_native_rgb.txt 2>&1
done
topdown.py -l3 -nd -os scan_file_rgb.txt -o results_topdown_rgb.txt -f ./dwt2d "rgb.bmp -d 1024x1024 -f -5 -l 3"