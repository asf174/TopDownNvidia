"""
Program that implements the Top Down methodology over NVIDIA GPUs.

@author:    Alvaro Saiz (UC)
@date:      Jan-2021
@version:   1.0
"""

import argparse
import sys
#from tabulate import tabulate #TODO, pintar
from errors.topdown_errors import *
from parameters.topdown_params import TopDownParameters # parameters of program
from measure_levels.level_one import LevelOne
from measure_levels.level_two import LevelTwo
from show_messages.message_format import MessageFormat
from args.unique_argument import DontRepeat

class TopDown:
    """
    Class that implements TopDown methodology over NVIDIA GPUs.
    
    Attributes:
        __level                         : int   ;   level of the exection
        __file_output                   : str   ;   path to log to show results or 'None' if option
                                                    is not specified
        __show_long_desc                : bool  ;   True to show long-descriptions of results or
                                                    False in other case
        __program                       : str   ;   path to program to be executed
        __delete_output_file_content    : bool  ;   If '-o/--output' is set delete output's file contents before 
                                                    write results
        __show_desc                     : bool  ;   True to show descriptions of results or
                                                    False in other case
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
        self.__delete_output_file_content : bool = args.delete_output_file_content
        self.__show_desc : bool = args.show_desc
        # introduction
        self.__intro_message()
        pass
    
    def __add_show_desc_argument(self, parser : argparse.ArgumentParser):
        """ 
        Add show-description argument. 'C_LONG_DESCRIPTION_SHORT_OPTION' is the short 
        option of argument and 'C_LONG_DESCRIPTION_LONG_OPTION' is the long version of argument.

        Params:
            parser : argparse.ArgumentParser ; group of the argument.
        """

        parser.add_argument (
            TopDownParameters.C_SHOW_DESCRIPTION_SHORT_OPTION, 
            TopDownParameters.C_SHOW_DESCRIPTION_LONG_OPTION, 
            help = 'description of results.',
            action = 'store_false',
            dest = 'show_desc')
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
            action = DontRepeat, # preguntar TODO
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
    
    def __add_delete_output_file_content(self, parser : argparse.ArgumentParser):
        """ 
        Add output's file delete content argument. 'C_DELETE_CONTENT_SHORT_OPTION' is the short 
        option of argument and 'C_DELETE_CONTENT_LONG_OPTION' is the long version of argument.

        Params:
            parser : argparse.ArgumentParser ; group of the argument.
        """

        parser.add_argument (
            TopDownParameters.C_DELETE_CONTENT_SHORT_OPTION, 
            TopDownParameters.C_DELETE_CONTENT_LONG_OPTION, 
            help = "If '-o/--output' is set delete output's file contents before write results",
            action = 'store_true',
            dest = 'delete_output_file_content')
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
        self.__add_delete_output_file_content(parser)
        self.__add_show_desc_argument(parser)
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

    def show_desc(self) -> bool:
        """
        Check if program has to show description of results.

        Returns:
            True to show description of False if not
        """

        return self.__show_desc
        pass

    def delete_output_file_content(self) -> bool:
        """
        Check if program has to delete output's file contents.

        Returns:
            True to delete output's file content or False if not
        """

        return self.__delete_output_file_content
        pass

    def __intro_message(self): 
        """ Intro message with information."""

        printer : MessageFormat = MessageFormat()
        printer.print_center_msg_box(msg = "TopDown Metholodgy over NVIDIA's GPUs", indent = 1, title = "", output_file = self.output_file(), 
        width = None, delete_content_file = self.delete_output_file_content())
        print()
        print()
        print()
        message : str = "\n\n\nWelcome to the " + sys.argv[0] + " program where you can check the bottlenecks of your\n" + \
            "CUDA program running on NVIDIA GPUs. This analysis is carried out considering the architectural\n" + \
            "aspects of your GPU, in its different parts. The objective is to detect the bottlenecks in your\n" + \
            "program, which cause the IPC (Instructions per Cycle) to be drastically reduced."
        printer.print_max_line_length_message(message, TopDownParameters.C_NUM_MAX_CHARACTERS_PER_LINE, self.output_file(), False)
        print()
        print()
        message = "\n\nNext, you can see general information about the program\n"
        printer.print_max_line_length_message(message, TopDownParameters.C_NUM_MAX_CHARACTERS_PER_LINE, self.output_file(), False)

        printer.print_msg_box(msg = TopDownParameters.C_INTRO_MESSAGE_GENERAL_INFORMATION, indent = 1, title = "GENERAL INFORMATION",
         output_file = self.output_file(), width = None, delete_content_file = False)
        print()
        message = "\n\nIn accordance with what has been entered, the execution will be carried out following the following terms:\n"
        printer.print_max_line_length_message(message, TopDownParameters.C_NUM_MAX_CHARACTERS_PER_LINE, self.output_file(), False)
        message = ("\n- Execution Level:                  " + str(self.level()) + "\n" + \
                   "- Analyzed program:                 " + self.program() + "\n" + \
                   "- Output File:                      " + self.output_file() + "\n" + \
                   "- Long-Description:                 " + str(self.show_long_desc()) + "\n" + \
                   "- Delete output's file content:     " + str(self.delete_output_file_content()))
        printer.print_msg_box(msg = message, indent = 1, title = "EXECUTION FEATURES", output_file = self.output_file(), width = None,
            delete_content_file = False)
        print()
        message = "\n\nSaid that, according to the level entered by you, WE START THE ANALYSIS.\n"
        printer.print_max_line_length_message(message = message, max_length = TopDownParameters.C_NUM_MAX_CHARACTERS_PER_LINE, output_file = self.output_file(), delete_content_file = False)
        print()
        pass
      
    def __show_results(self, level_execution):
        """ Show Results of execution indicated by argument.

        Parameters:
            level_one   : LevelOne/LevelTwo  ; reference to level one/two analysis ALREADY DONE.
        """

        # check type of level execution
        if type(level_execution) is LevelOne:
            level_execution.__class__ = LevelOne
        elif type(level_execution) is LevelTwo:
            level_execution.__class__ = LevelTwo
        else:
            print("ERROR")
        print()
        printer : MessageFormat = MessageFormat()
        message : str = "\nThe results have been obtained correctly. General results of IPC are the following:\n\n"
        printer.print_max_line_length_message(message = message, max_length = TopDownParameters.C_NUM_MAX_CHARACTERS_PER_LINE, output_file = self.output_file(), delete_content_file = False)
        print()
        message = "IPC OBTAINED: " + str(level_execution.ipc()) + " | MAXIMUM POSSIBLE IPC: " +  str(level_execution.get_device_max_ipc())
        printer.print_desplazed_msg_box(msg = message, indent = 1, title = "", output_file = self.output_file(), width = None, delete_content_file = False)
        print()
        message = ("\n\n'IPC OBTAINED' is the IPC of the analyzed program (computed by scan tool) and 'MAXIMUM POSSIBLE IPC'\n" +
            "is the the maximum IPC your GPU can achieve. This is computed taking into account architectural concepts, such as the\n" +
            "number of warp planners per SM, as well as the number of Dispatch units of each SM.")
        printer.print_max_line_length_message(message = message, max_length = TopDownParameters.C_NUM_MAX_CHARACTERS_PER_LINE, output_file = self.output_file(), delete_content_file = False)
        message = ("\n    As you can see, the IPC obtanied it " + "is " + str(round((level_execution.get_device_max_ipc()/level_execution.ipc())*100, 2)) + 
            "% smaller than you could get. This lower IPC is due to STALLS in the different \nparts of the architecture and DIVERGENCE problems. " +
            "We analyze them based on the level of the TopDown:\n")
        printer.print_max_line_length_message(message = message, max_length = TopDownParameters.C_NUM_MAX_CHARACTERS_PER_LINE, output_file = self.output_file(), delete_content_file = False)
        print()  
        if self.show_desc():
            message = "\n" + level_execution.front_end().name() + ": " + level_execution.front_end().description() + "\n\n"
            printer.print_max_line_length_message(message = message, max_length = TopDownParameters.C_NUM_MAX_CHARACTERS_PER_LINE, output_file = self.output_file(), delete_content_file = False)
            print()
        message = ("{:<35} {:<5}".format('\nSTALLS, on the total (%):', str(level_execution.get_front_end_stall()) + '%\n\n'))
        message += ("{:<34} {:<5}".format('IPC DEGRADATION (%):', str(round(level_execution.front_end_percentage_ipc_degradation(), 3)) + '%\n'))
        printer.print_desplazed_msg_box(msg = message, indent = 1, title = level_execution.front_end().name() + " RESULTS", 
            output_file = self.output_file(), width = None, delete_content_file = False)
        if type(level_execution) is LevelTwo:
            if self.show_desc():
                message = "\n" + level_execution.front_band_width().name() + ": " + level_execution.front_band_width().description() + "\n\n"
                printer.print_max_line_length_message(message = message, max_length = TopDownParameters.C_NUM_MAX_CHARACTERS_PER_LINE, 
                output_file = self.output_file(), delete_content_file = False)
            print()
            message = ("{:<35} {:<5}".format('\nSTALLS, on the total (%):', str(level_execution.get_front_band_width_stall()) + '%\n\n'))
            message += ("{:<34} {:<5}".format('IPC DEGRADATION (%):', str(round(level_execution.front_band_width_percentage_ipc_degradation(), 3)) + '%\n'))
            printer.print_desplazed_msg_box(msg = message, indent = 1, title = level_execution.front_band_width().name() + " RESULTS", 
                output_file = self.output_file(), width = None, delete_content_file = False)
            print()
            if self.show_desc():
                message = "\n" + level_execution.front_dependency().name() + ": " + level_execution.front_dependency().description() + "\n\n"
                printer.print_max_line_length_message(message = message, max_length = TopDownParameters.C_NUM_MAX_CHARACTERS_PER_LINE, 
                output_file = self.output_file(), delete_content_file = False)
                print()
            message = ("{:<35} {:<5}".format('\nSTALLS, on the total (%):', str(level_execution.get_front_dependency_stall()) + '%\n\n'))
            message += ("{:<34} {:<5}".format('IPC DEGRADATION (%):', str(round(level_execution.front_dependency_percentage_ipc_degradation(), 3)) + '%\n'))
            message += "\n"
            printer.print_desplazed_msg_box(msg = message, indent = 1, title = level_execution.front_dependency().name() + " RESULTS", 
                output_file = self.output_file(), width = None, delete_content_file = False)
            print()
        if self.show_desc():
            message = "\n" + level_execution.back_end().name() + ": " + level_execution.back_end().description() + "\n\n"
            printer.print_max_line_length_message(message = message, max_length = TopDownParameters.C_NUM_MAX_CHARACTERS_PER_LINE, output_file = self.output_file(), 
            delete_content_file = False)
            print()
        message = ("{:<35} {:<5}".format('\nSTALLS, on the total (%):', str(level_execution.get_back_end_stall()) + '%\n\n'))
        message += ("{:<34} {:<5}".format('IPC DEGRADATION (%):', str(round(level_execution.back_end_percentage_ipc_degradation(), 3)) + '%\n'))
        printer.print_desplazed_msg_box(msg = message, indent = 1, title = level_execution.back_end().name() + " RESULTS", 
            output_file = self.output_file(), width = None, delete_content_file = False)
        print()
        if type(level_execution) is LevelTwo:
            if self.show_desc():
                message = "\n" + level_execution.memory_bound().name() + ": " + level_execution.memory_bound().description() + "\n\n"
                printer.print_max_line_length_message(message = message, max_length = TopDownParameters.C_NUM_MAX_CHARACTERS_PER_LINE, 
                output_file = self.output_file(), delete_content_file = False)
                print()
            message = ("{:<35} {:<5}".format('\nSTALLS, on the total (%):', str(level_execution.get_memory_bound_stall()) + '%\n\n'))
            message += ("{:<34} {:<5}".format('IPC DEGRADATION (%):', str(round(level_execution.memory_bound_percentage_ipc_degradation(), 3)) + '%\n'))
            printer.print_desplazed_msg_box(msg = message, indent = 1, title = level_execution.memory_bound().name() + " RESULTS", 
                output_file = self.output_file(), width = None, delete_content_file = False)
            print()
            if self.show_desc():
                message = "\n" + level_execution.core_bound().name() + ": " + level_execution.core_bound().description() + "\n\n"
                printer.print_max_line_length_message(message = message, max_length = TopDownParameters.C_NUM_MAX_CHARACTERS_PER_LINE, 
                output_file = self.output_file(), delete_content_file = False)
                print()
            message = ("{:<35} {:<5}".format('\nSTALLS, on the total (%):', str(level_execution.get_core_bound_stall()) + '%\n\n'))
            message += ("{:<34} {:<5}".format('IPC DEGRADATION (%):', str(round(level_execution.core_bound_percentage_ipc_degradation(), 3)) + '%\n'))
            printer.print_desplazed_msg_box(msg = message, indent = 1, title = level_execution.core_bound().name() + " RESULTS", 
                output_file = self.output_file(), width = None, delete_content_file = False)
            print()
        if self.show_desc():
            message = "\n" + level_execution.divergence().name() + ": " + level_execution.divergence().description() + "\n\n"
            printer.print_max_line_length_message(message = message, max_length = TopDownParameters.C_NUM_MAX_CHARACTERS_PER_LINE, 
            output_file = self.output_file(), delete_content_file = False)
            print()
        #message = ("{:<35} {:<5}".format('\nSTALLS, on the total (%):', str(level_one.get_divergence_stall()) + '%\n\n'))
        message = ("{:<34} {:<5}".format('\nIPC DEGRADATION (%):', str(round(level_execution.divergence_percentage_ipc_degradation(), 3)) + '%\n'))
        printer.print_desplazed_msg_box(msg = message, indent = 1, title = level_execution.divergence().name() + " RESULTS", 
            output_file = self.output_file(), width = None, delete_content_file = False)
        print()
        pass

    def launch(self):
        """ Launch execution."""
        
        if self.level() == 1:
           level : LevelOne = LevelOne(self.program(), self.output_file())
        else:
            level : LevelTwo = LevelTwo(self.program(), self.output_file()) 
        lst_output : list[str] = list() # for extra information
        level.run(lst_output)
        self.__show_results(level)
        if self.show_desc() and self.show_long_desc(): ### revisar este IF no deberia ir aqui
            # Write results in output-file if has been specified
            if not self.output_file() is None:
                MessageFormat().write_in_file_at_end(self.output_file(), lst_output)
            element : str
            for element in lst_output:
                print(element)
    
if __name__ == '__main__':
    td = TopDown()
    td.launch()
    MessageFormat().print_max_line_length_message(message = "Analysis performed correctly!", 
    max_length = TopDownParameters.C_NUM_MAX_CHARACTERS_PER_LINE, output_file = td.output_file(), delete_content_file = False)