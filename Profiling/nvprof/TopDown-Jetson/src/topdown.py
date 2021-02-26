"""
Program that implements the Top Down methodology over NVIDIA GPUs.

@author:    Alvaro Saiz (UC)
@date:      Jan-2021
@version:   1.0
"""

import argparse
import re
#from tabulate import tabulate #TODO, pintar
from errors.topdown_errors import *
from measure_parts.extra_measure import ExtraMeasure    
from shell.shell import Shell # launch shell arguments
from parameters.topdown_params import TopDownParameters # parameters of program
import sys
from measure_levels.level_one import LevelOne
from show_messages.message_format import MessageFormat
from args.unique_argument import DontRepeat

class TopDown:
    """
    Class that implements TopDown methodology over NVIDIA GPUs.
    
    Attributes:
        __level             : int   ;   level of the exection
        __file_output       : str   ;   path to log to show results or 'None' if option
                                        is not specified
        __show_long_desc    : bool  ;   True to show long-descriptions of results or
                                        false in other case
        __program           : str   ;   path to program to be executed
    """

    def __init__(self):
        """
        Init attributes depending of arguments.
        """

        parser : argparse.ArgumentParse = argparse.ArgumentParser(#prog='[/path/to/PROGRAM]',
            formatter_class = lambda prog: argparse.HelpFormatter(prog, max_help_position = 50),
            description = "TopDown methodology on NVIDIA's GPUs",
            epilog = "Check options to run program")
            #usage='%(prog)s [OPTIONS]') #exit_on_error=False)
        parser._optionals.title = "Optional Arguments"
        self.__add_arguments(parser)

        # Save values into attributes
        args : argparse.Namespace = parser.parse_args()
      
        self.__level : int = args.level[0]
        self.__file_output : str = args.file
        self.__show_long_desc : bool = args.desc
        self.__program : str = args.program

        # introduction
        self.__intro_message()
        pass
    
    def __add_program_argument(self, group : argparse._ArgumentGroup) :
        """ 
        Add program argument. 'C_FILE_SHORT_OPTION' is the short option of argument
        and 'C_FILE_LONG_OPTION' is the long version of argument.

        Params:
            group : argparse._ArgumentGroup ; group of the argument.
        """

        group.add_argument(
            TopDownParameters.C_FILE_SHORT_OPTION, 
            TopDownParameters.C_FILE_LONG_OPTION, 
            required = True,
            help = 'run file. Path to file.',
            default = None,
            nargs = '?', 
            action = DontRepeat,
            type = str, 
            #metavar='/path/to/file',
            dest = 'program')
        pass

    def __add_level_argument(self, group : argparse._ArgumentGroup):
        """ 
        Add level argument. 'C_LEVEL_SHORT_OPTION' is the short option of argument
        and 'C_LEVEL_LONG_OPTION' is the long version of argument.

        Params:
            group : argparse._ArgumentGroup ; group of the argument.
        """
        
        group.add_argument (
            TopDownParameters.C_LEVEL_SHORT_OPTION, TopDownParameters.C_LEVEL_LONG_OPTION,
            required = True,
            help = 'level of execution.',
            type = int,
            action = DontRepeat,
            nargs = 1,
            default = -1,
            choices = range(TopDownParameters.C_MIN_LEVEL_EXECUTION, TopDownParameters.C_MAX_LEVEL_EXECUTION + 1), # range [1,2], produces error, no if needed
            metavar = '[NUM]',
            dest = 'level')
        pass

    def __add_ouput_file_argument(self, parser : argparse.ArgumentParser):
        """ 
        Add ouput-file argument. 'C_OUTPUT_FILE_SHORT_OPTION' is the short option of argument
        and 'C_OUTPUT_FILE_LONG_OPTION' is the long version of argument.

        Params:
            parser : argparse.ArgumentParser ; group of the argument.
        """
        
        parser.add_argument (
            TopDownParameters.C_OUTPUT_FILE_SHORT_OPTION, 
            TopDownParameters.C_OUTPUT_FILE_LONG_OPTION, 
            help = 'output file. Path to file.',
            default = None,
            nargs = '?', 
            type = str, 
            #metavar='/path/to/file',
            dest = 'file')
        pass

    def __add_long_desc_argument(self, parser : argparse.ArgumentParser):
        """ 
        Add long-description argument. 'C_LONG_DESCRIPTION_SHORT_OPTION' is the short 
        option of argument and 'C_LONG_DESCRIPTION_LONG_OPTION' is the long version of argument.

        Params:
            parser : argparse.ArgumentParser ; group of the argument.
        """

        parser.add_argument (
            TopDownParameters.C_LONG_DESCRIPTION_SHORT_OPTION, 
            TopDownParameters.C_LONG_DESCRIPTION_LONG_OPTION, 
            help = 'long description of results.',
            action = 'store_true',
            dest = 'desc')
        pass
    
    def __add_arguments(self, parser : argparse.ArgumentParser):
        """ 
        Add arguments of the pogram.

        Params:
            parser : argparse.ArgumentParser ; group of the arguments.
        """

        # Create group for required arguments
        requiredGroup : argparse._ArgumentGroup = parser.add_argument_group("Required arguments")

        self.__add_program_argument(requiredGroup)
        self.__add_level_argument(requiredGroup)
        self.__add_ouput_file_argument(parser)
        self.__add_long_desc_argument(parser)
        pass

    def program(self) -> str:
        """
        Returns path to runnable program.

        Returns:
            str with path to program to be executed
        """
        return self.__program
        pass
    
    def level(self) -> int:
        """ 
        Find the TopDown run level.

        Returns:
            1 if it's level one, 2 if it's level two
        """ 
        return self.__level
        pass

    def output_file(self) -> str:
        """
        Find path to output file.

        Returns:
            path to file to write, or None if 
            option '-o' or '--output' has not been indicated
        """
        return self.__file_output # descriptor to file or None
        pass
    
    def show_long_desc(self) -> bool:
        """
        Check if program has to show long description of results.

        Returns:
            True to show long description of False if not
        """

        return self.__show_long_desc
        pass

    def __intro_message(self): 
        """ Intro message with information."""

        printer : MessageFormat = MessageFormat()
        printer.print_center_msg_box(msg = "TopDown Metholodgy over NVIDIA's GPUs", indent = 1, title = "", output_file = self.output_file(), width = None)
        print()
        print()
        print()
        message : str = "\n\n\nWelcome to the " + sys.argv[0] + " program where you can check the bottlenecks of your \n" + \
            "CUDA program running on NVIDIA GPUs. This analysis is carried out considering the architectural \n" + \
            "aspects of your GPU, in its different parts. The objective is to detect the bottlenecks in your \n" + \
            "program, which cause the IPC (Instructions per Cycle) to be drastically reduced."
        printer.print_max_line_length_message(message, 130, self.output_file())
        print()
        print()
        message = "\n\nNext, you can see general information about the program\n"
        printer.print_max_line_length_message(message, 130, self.output_file())

        message = ("\n- Program Name:    topdown.py\n" + \
                   "- Author:          Alvaro Saiz (UC)\n" + \
                   "- Contact info:    asf174@alumnos.unican.es\n" + \
                   "- Company:         University Of Cantabria\n" + \
                   "- Place:           Santander, Cantabria, Kingdom of Spain\n" + \
                   "- Teachers:        Pablo Abad (UC) <pablo.abad@unican.es>, Pablo Prieto (UC) <pablo.prieto@unican.es>\n" + \
                   "- Bugs Report:     asf174@alumnos.unican.es"+ 
                   "\n\n- Licence:         GNU GPL")
        printer.print_msg_box(msg = message, indent = 1, title = "GENERAL INFORMATION", output_file = self.output_file(), width = None)
        print()
        message = "\n\nIn accordance with what has been entered, the execution will be carried out following the following terms:\n"
        printer.print_max_line_length_message(message, 130, self.output_file())
        message = ("\n- Execution Level:     " + str(self.level()) + "\n" + \
                   "- Analyzed program:    " + self.program() + "\n" + \
                   "- Output File:         " + self.output_file() + "\n" + \
                   "- Long-Description:    " + str(self.show_long_desc()) )
        printer.print_msg_box(msg = message, indent = 1, title = "PROGRAM FEATURES", output_file = self.output_file(), width = None)
        print()
        message = "\n\nSaid that, according to the level entered by you, WE START THE ANALYSIS.\n"
        printer.print_max_line_length_message(message = message, max_length = 130, output_file = self.output_file())
        print()
        pass

    def _show_level_one_results(self, level_one : LevelOne):
        """ Show Results of level one of execution indicated by argument.

        Parameters:
            level_one   : LevelOne  ; reference to level-one analysis ALREADY DONE.
        """
        print()
        printer : MessageFormat = MessageFormat()
        message : str = "\nThe results have been obtained correctly. General results of IPC are the following:\n\n"
        printer.print_max_line_length_message(message = message, max_length = 130, output_file = self.output_file())
        print()
        message = "IPC OBTAINED: " + str(level_one.ipc()) + " | MAXIMUM POSSIBLE IPC: " +  str(level_one.get_device_max_ipc())
        printer.print_desplazed_msg_box(msg = message, indent = 1, title = "", output_file = self.output_file(), width = None)
        print()
        message = ("\n\n'IPC OBTAINED' is the IPC of the analyzed program (computed by scan tool) and 'MAXIMUM POSSIBLE IPC'\n" +
            "is the the maximum IPC your GPU can achieve. This is computed taking into account architectural concepts, such as the \n" +
            "number of warp planners per SM, as well as the number of Dispatch units of each SM.")
        printer.print_max_line_length_message(message = message, max_length = 130, output_file = self.output_file())
        message = ("    \n\nAs you can see, the IPC obtanied it " + "is " + str(round((level_one.get_device_max_ipc()/level_one.ipc())*100, 2)) + 
            "% smaller than you could get. This lower IPC is due to STALLS in the different \nparts of the architecture and DIVERGENCE problems. " +
            "We analyze them based on the level of the TopDown:\n")
        printer.print_max_line_length_message(message = message, max_length = 130, output_file = self.output_file())
        print()
        
        message = "\n\n" + level_one.front_end().name() + ": " + level_one.front_end().description() + "\n\n"
        printer.print_max_line_length_message(message = message, max_length = 130, output_file = self.output_file())
        print()
        
        message = ("{:<35} {:<5}".format('\nSTALLS, on the total (%):', str(level_one.get_front_end_stall()) + '%\n\n'))
        message += ("{:<34} {:<5}".format('IPC DEGRADATION (%):', str(round(level_one.front_end_percentage_ipc_degradation(), 3)) + '%\n'))
        printer.print_desplazed_msg_box(msg = message, indent = 1, title = level_one.front_end().name() + " RESULTS", 
            output_file = self.output_file(), width = None)
        print()

        message = "\n\n" + level_one.back_end().name() + ": " + level_one.back_end().description() + "\n\n"
        printer.print_max_line_length_message(message = message, max_length = 130, output_file = self.output_file())
        print()
        
        message = ("{:<35} {:<5}".format('\nSTALLS, on the total (%):', str(level_one.get_back_end_stall()) + '%\n\n'))
        message += ("{:<34} {:<5}".format('IPC DEGRADATION (%):', str(round(level_one.back_end_percentage_ipc_degradation(), 3)) + '%\n'))
        printer.print_desplazed_msg_box(msg = message, indent = 1, title = level_one.back_end().name() + " RESULTS", 
            output_file = self.output_file(), width = None)
        print()

        message = "\n\n" + level_one.divergence().name() + ": " + level_one.divergence().description() + "\n\n"
        printer.print_max_line_length_message(message = message, max_length = 130, output_file = self.output_file())
        print()
        
        #message = ("{:<35} {:<5}".format('\nSTALLS, on the total (%):', str(level_one.get_divergence_stall()) + '%\n\n'))
        message = ("{:<34} {:<5}".format('\nIPC DEGRADATION (%):', str(round(level_one.divergence_percentage_ipc_degradation(), 3)) + '%\n'))
        printer.print_desplazed_msg_box(msg = message, indent = 1, title = level_one.divergence().name() + " RESULTS", 
            output_file = self.output_file(), width = None)
        print()
    
        pass

    def level_one(self):
        """ 
        Run TopDown level 1.
        """

        level_one : LevelOne = LevelOne(self.program(), self.output_file())
        lst_output : list[str] = list()
        level_one.run(lst_output)
        self._show_level_one_results(level_one)
        if self.show_long_desc(): ### revisar este IF no deberia ir aqui
            # Write results in output-file if has been specified
            if not self.output_file() is None:
                MessageFormat().write_in_file_at_end(self.output_file(), lst_output)
            element : str
            for element in lst_output:
                print(element)
        pass

    def level_two(self):
        """ 
        Run TopDown level 2.
        """
        # TODO
        pass
    
if __name__ == '__main__':
    td = TopDown()
    level : int = td.level()

    if level == 1:
        td.level_one()
    elif level == 2:
        td.level_two()
    print()
    print("Analysis performed correctly!")