import sys
import pathlib
import os.path
import numpy as np
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Grafica:
     # Arguments Features
    MIN_NUMBER_ARGUMENTS = 6
    MAX_NUMBER_ARGUMENTS = 8
    NUMBER_ARGUMENT_WITH_SOME_LIMITS = 7
    NUMBER_ARGUMENT_WITH_ALL_LIMITS = 8
    HELP_OPTION = "help"
 

    def _addFeatures():
        # Add title and axis names
        plt.title(sys.argv[1])
        plt.xlabel(sys.argv[2])
        plt.ylabel(sys.argv[3])
        pass

    def _addData():
        # colours for graphs
        colourScatter = ['blue', 'red', 'orange', 'grey']
        colourPlot = ['blue', 'red', 'orange', 'grey']
        
        # introduce fileNames in array
        files = sys.argv[4]    
        files = argument.replace('[','')
        files = argument.replace(']','')
        files = argument.split(',')

        x_axis = []
        y_axis = []
        count = 0
        colourNumber = 0
        for fileName in files:
            # read file
            f = open(fileName, 'r') 
            Lines = f.readlines() 
            
            # add data
            for line in Lines: 
                lineList = line.split()
                
                for word in lineList:
                    if count == 0:
                        x_axis.append(word)
                        count = count + 1
                    else:
                        y_axis.append(word)
                        count = 0
            plt.scatter(x_axis,y_axis,color=colourScatter[colourNumber],zorder=1)
            plt.plot(x_axis,y_axis,color=colourPlot[colourNumber],zorder=2)
            colourNumber = colourNumber + 1
            x_axis = []
            y_axis = []
        pass

    def _check_arguments(self):
        MIN_NUMBER_ARGUMENTS = 6
        if len(sys.argv) < MIN_NUMBER_ARGUMENTS or len(sys.argv) > MAX_NUMBER_ARGUMENTS:
            print("Entro aki")
            if len(sys.argv) == 1:
                print("Error with number arguments")
            elif sys.argv[1] == HELP_OPTION:
                print("python3 " + sys.argv[0] + " <title> <titleX> <titleY> [<file[0]>,<file[n]>] <nameFig> [<xLimStart>,<xLimEnd>,<XStep>], [<yLimStart>,<yLimEnd>,<Ystep>]")
            sys.exit()
        pass

    def _check_limits():
        if len(sys.argv) == NUMBER_ARGUMENT_WITH_SOME_LIMITS:
            argument = sys.argv[6]
            type_axis = argument[0]
            
            argument = argument.replace(argument[0],'')
            argument = argument.replace('[','')
            argument = argument.replace(']','')
            argument = argument.split(',')
            
            if (type_axis == 'y') or (type_axis == 'Y'):
                plt.ylim(int(argument[0]),int(argument[1]))
                plt.yticks( np.arange(int(argument[0]), int(argument[1]), step=int(argument[2])))
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
        pass


    def main(self):
        _check_arguments()
        _addFeatures()
        _addData()
        
        _check_limits()

        # show
        plt.grid()
        plt.show()

        # save
        plt.savefig(sys.argv[5])


Grafica().main()