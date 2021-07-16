#!/usr/bin/python3
# Medium of 'real' time of time command, deleting MAX and MIN (medium between the other)
import sys
import statistics
# Using readlines()
#print(sys.argv[2])
file1 = open(sys.argv[1], 'r')
Lines = file1.readlines()
 
t_min = 0.0
t_sec = 0.0
l_result = list()
# Strips the newline character
for line in Lines:
    words = list(line.split("\t"))
    #print(words)
    if words[0] == "real":
        #print("ENTRO")
        number = words[len(words) - 1]
        l_real = number.partition('m')
        minut = float(l_real[0])
        sec = l_real[len(l_real) - 1]
        sec = l_real[2][:-1].partition('s')[0]
        sec = sec.replace(',','.')
        sec = float(sec)
        #print(minut)
        #print(sec)
        l_result.append(minut*60 + sec)
#print(l_result)
l_result.remove(max(l_result))
l_result.remove(min(l_result))
print(str(statistics.mean(l_result)) + " sec")
