import sys
import pathlib
import os.path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



# Arguments Features
NUMBER_ARGUMENTS = 6
NUMBER_ARGUMENT_WITH_SOME_LIMITS = 7
NUMBER_ARGUMENT_WITH_ALL_LIMITS = 8
HELP_OPTION = "help"


if (len(sys.argv) != NUMBER_ARGUMENTS or len(sys.argv) != NUMBER_ARGUMENT_WITH_LIMITS 
    or len(sys.argv) != NUMBER_ARGUMENT_WITH_ALL_LIMITS):
    if len(sys.argv) == 1:
        print("Error with number arguments")
    elif sys.argv[1] == HELP_OPTION:
        print("python3 " + sys.argv[0] + " <title> <titleX> <titleY> <file> <nameFig> [<xLimStart>,<xLimEnd>,<XStep>]," +
        + "[<yLimStart>,<yLimEnd>,<Ystep>]")
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

## check limits
if len(sys.argv) == NUMBER_ARGUMENT_WITH_SOME_LIMITS:
    argument = sys.argv[6]
    type_axis = argument[0]
    
    argument = argument.replace(argument[0],'')
    argument = argument.replace('[','')
    argument = argument.replace(']','')
    argument = argument.split(',')
    
    if type_axis == 'y' or type_axis == 'Y':
        plt.ylim(int(argument[0]),int(argument[1]))
        plt.yticks(np.arange(int(argument[0]), int(argument[1]), step=int(argument[2]))
    elif type_axis == 'X' or type_axis == 'x':
        plt.xlim(int(argument[0]), int(argument[1]))
        plt.xticks(np.arange(int(argument[0]), int(argument[1]), step=int(argument[2])))
    else:
        print("ERROR")

elif len(sys.argv) == NUMBER_ARGUMENT_WITH_ALL_LIMITS:
    ## X limits
    argument = sys.argv[6]
    type_axis = argument[0]
    
    argument = argument.replace(argument[0],'')
    argument = argument.replace('[','')
    argument = argument.replace(']','')
    argument = argument.split(',')

    plt.xlim(int(argument[0]), int(argument[1]))
    plt.xticks(np.arange(int(argument[0]), int(argument[1]), step=int(argument[2])))
    
    ## Y limits
    argument = sys.argv[7]
    type_axis = argument[0]
    
    argument = argument.replace(argument[0],'')
    argument = argument.replace('[','')
    argument = argument.replace(']','')
    argument = argument.split(',')

    plt.ylim(int(argument[0]), int(argument[1]))
    plt.yticks(np.arange(int(argument[0]), int(argument[1]), step=int(argument[2])))    

# print and show
plt.scatter(x_axis,y_axis,color='r',zorder=1)
plt.plot(x_axis,y_axis,color='b',zorder=2)
plt.grid()
plt.show()

# save
plt.savefig(sys.argv[5])