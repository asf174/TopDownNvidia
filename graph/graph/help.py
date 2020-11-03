import sys
import pathlib
import os.path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NUMBER_ARGUMENT_WITH_SOME_LIMITS = 2
NUMBER_ARGUMENT_WITH_ALL_LIMITS = 3

if len(sys.argv) == NUMBER_ARGUMENT_WITH_SOME_LIMITS:
    argument = sys.argv[1]

    type = argument[0]
    argument = argument.replace(argument[0],'')
    argument = argument.replace('[','')
    argument = argument.replace(']','')
    argument = argument.split(',')
    if type == 'y' or type == 'Y':
        print("He leido Y")
    elif argument[0] == 'X' or type == 'x':
        print("He leido X")
    else:
        print("ERROR")

elif len(sys.argv) == NUMBER_ARGUMENT_WITH_ALL_LIMITS:
    print("No implementado")
