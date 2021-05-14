"""
Program who filters the results obtained from topdown.py
and show the results.

@author Alvaro Saiz (UC)
@date May 2021
@version 1.0
"""

import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(1, parentdir)
from parameters.parser_parameters import ParserParamters
from parameters.topdown_parameters import TopDownParameters

class Parser:
    """ 
    Class which filters the results obtained from topdown.pt
    and show the results.
    """

    def __init__(self):
        """ 
        Init attributes depending of arguments.
        """

        self.__parser : argparse.ArgumentParse = argparse.ArgumentParser(
            formatter_class = lambda prog : argparse.HelpFormatter(prog, max_help_position = 50),
            description = "Parser results obtainded from topdown.py")
        self.__parser_optionals.title = "Optional Arguments"
        self.__add_arguments(self.__parser)

        # Save values
        args : argparse.Namespace = self.__parser.parse_args()

        pass

    def __add_input_file_argument(self, group : argparse._ArgumentGroup): 
        """ 
        Add input-file argument. 'C_INPUT_FILE_ARGUMENT_SHORT_OPTION' is the short option of argument
        and 'C_INPUT_FILE_ARGUMENT_LONG_OPTION' is the long version of argument.

        Params:
            group : argparse._ArgumentGroup ; group of the argument.
        """
        
        group.add_argument(
            ParserParameters.C_INPUT_FILE_ARGUMENT_SHORT_OPTION,
            ParserParameters.C_INPUT_FILE_ARGUMENT_LONG_OPTION,
            required = True,
            help = TopDownParameters.C_INPUT_FILE_ARGUMENT_DESCRIPTION,
            default = None,
            nargs = '?',  
            action = DontRepeat,
            type = str,
            #metavar='/path/to/file',
            dest = 'input_file')
        pass
    
     def __add_level_argument(self, group : argparse._ArgumentGroup):
        """ 
        Add level argument. 'C_LEVEL_ARGUMENT_SHORT_OPTION' is the short option of argument
        and 'C_LEVEL_ARGUMENT_LONG_OPTION' is the long version of argument.

        Params:
            group : argparse._ArgumentGroup ; group of the argument.
        """
                                                      
        group.add_argument(
            TopDownParameters.C_LEVEL_ARGUMENT_SHORT_OPTION, TopDownParameters.C_LEVEL_ARGUMENT_LONG_OPTION,
            required = True,
            help = TopDownParameters.C_LEVEL_ARGUMENT_DESCRIPTION,
            type = int,
            action = DontRepeat,
            nargs = 1,
            default = -1, 
            choices = range(TopDownParameters.C_MIN_LEVEL_EXECUTION, TopDownParameters.C_MAX_LEVEL_EXECUTION + 1), # range [1,3], produces error, no if needed
            metavar = '[NUM]',
            dest = 'level')
        pass


