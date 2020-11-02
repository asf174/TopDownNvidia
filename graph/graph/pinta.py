import sys
import pathlib
import os.path
import numpy as np
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt



# Arguments Features
NUMBER_ARGUMENTS = 6
HELP_OPTION = "help"


if len(sys.argv) != NUMBER_ARGUMENTS:
    if len(sys.argv) == 1:
        print("Error with number arguments")
        sys.exit()
    elif sys.argv[1] == HELP_OPTION:
        print("python3 " + sys.argv[0] + " <title> <titleX> <titleY> <file> <nameFig>")
        sys.exit()

# Add title and axis names
plt.title(sys.argv[1])
plt.xlabel(sys.argv[2])
plt.ylabel(sys.argv[3])


# read file
file1 = open(sys.argv[4], 'r') 
Lines = file1.readlines() 
  
count = 0
x_axis = []
y_axis = []

# add data
for line in Lines: 
   line = line.split()
   for word in line:
    if count == 0:
        x_axis.append(word)
        count = count + 1
    else:
        y_axis.append(word)
        count = 0

plt.plot(x_axis,y_axis)

plt.grid()		
plt.show()
#plt.savefig('file')
print("He llegado aqui")