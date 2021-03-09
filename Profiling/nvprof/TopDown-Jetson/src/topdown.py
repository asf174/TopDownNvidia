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
from measure_levels.level_three import LevelThree
from show_messages.message_format import MessageFormat
from args.unique_argument import DontRepeat

class TopDown:
    """
    Class that implements TopDown methodology over NVIDIA GPUs.
    
    Attributes:
        __level                         : int   ;   level of the exection
        __file_output                   : str   ;   path to log to show results or 'None' if option
                                                    is not specified
        __verbose                       : bool  ;   True to show long-descriptions of results or
                                                    False in other case
        __program                       : str   ;   path to program to be executed
        __delete_output_file_content    : bool  ;   If '-o/--output' is set delete output's file contents before 
                                                    write results
        __show_desc                     : bool  ;   True to show descriptions of results or
                                                    False in other case
        __show_metrics                  : bool  ;   True if program has to show metrics computed by NVIDIA scan tool
        __show_events                   : bool  ;   True if program has to show events computed by NVIDIA scan tool
        __show_all_measurementes        : bool  ;   True if program has to show all measures (metrics and events) 
                                                    computed by NVIDIA scan tool
        __is_tesla_device               : bool  ;   True if device is a tesla model or False if not
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
        self.__verbose : bool = args.verbose
        self.__program : str = args.program
        self.__delete_output_file_content : bool = args.delete_output_file_content
        self.__show_desc : bool = args.show_desc
        self.__show_metrics : bool = args.metrics
        self.__show_events : bool = args.events
        self.__show_all_measurementes : bool = args.all_measures
        self.__is_tesla_device : bool = args.tesla
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
            help = "don't show description of results.",
            action = 'store_false',
            dest = 'show_desc')
        pass

    def __add_metrics_argument(self, parser : argparse.ArgumentParser):
        """ 
        Add tesla argument. 'C_METRICS_SHORT_OPTION' is the short option of argument
        and 'C_METRICS_LONG_OPTION' is the long version of argument.

        Params:
            parser : argparse.ArgumentParser ; group of the argument.
        """

        parser.add_argument (
            TopDownParameters.C_METRICS_SHORT_OPTION, 
            TopDownParameters.C_METRICS_LONG_OPTION, 
            help = 'show metrics computed by NVIDIA scan tool',
            action = 'store_true',
            dest = 'metrics')
        pass


    def __add_tesla_argument(self, parser : argparse.ArgumentParser):
        """ 
        Add tesla argument. 'C_TESLA_DEVICE_SHORT_OPTION' is the short option of argument
        and 'C_TESLA_DEVICE_LONG_OPTION' is the long version of argument.

        Params:
            parser : argparse.ArgumentParser ; group of the argument.
        """

        parser.add_argument (
            TopDownParameters.C_TESLA_DEVICE_SHORT_OPTION, 
            TopDownParameters.C_TESLA_DEVICE_LONG_OPTION, 
            help = 'set if device used is a tesla model',
            action = 'store_true',
            dest = 'tesla')
        pass

    def __add_all_measures_argument(self, parser : argparse.ArgumentParser):
        """ 
        Add all measures argument. 'C_ALL_MEASURES_SHORT_OPTION' is the short option of argument
        and 'C_ALL_MEASURES_LONG_OPTION' is the long version of argument.

        Params:
            parser : argparse.ArgumentParser ; group of the argument.
        """

        parser.add_argument (
            TopDownParameters.C_ALL_MEASURES_SHORT_OPTION, 
            TopDownParameters.C_ALL_MEASURES_LONG_OPTION, 
            help = 'show all measures (events and metrics) computed by NVIDIA scan tool',
            action = 'store_true',
            dest = 'all_measures')
        pass

    def __add_events_argument(self, parser : argparse.ArgumentParser):
        """ 
        Add events argument. 'C_EVENTS_SHORT_OPTION' is the short option of argument
        and 'C_EVENTS_LONG_OPTION' is the long version of argument.

        Params:
            parser : argparse.ArgumentParser ; group of the argument.
        """

        parser.add_argument (
            TopDownParameters.C_EVENTS_SHORT_OPTION, 
            TopDownParameters.C_EVENTS_LONG_OPTION, 
            help = 'show events computed by NVIDIA scan tool',
            action = 'store_true',
            dest = 'events')
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

    def __add_verbose_argument(self, parser : argparse.ArgumentParser):
        """ 
        Add verbose argument. 'C_VERBOSE_SHORT_OPTION' is the short 
        option of argument and 'C_VERBOSE_LONG_OPTION' is the long version of argument.

        Params:
            parser : argparse.ArgumentParser ; group of the argument.
        """

        parser.add_argument (
            TopDownParameters.C_VERBOSE_SHORT_OPTION, 
            TopDownParameters.C_VERBOSE_LONG_OPTION, 
            help = 'long description of results.',
            action = 'store_true',
            dest = 'verbose')
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
        self.__add_verbose_argument(parser)
        self.__add_delete_output_file_content(parser)
        self.__add_show_desc_argument(parser)
        self.__add_metrics_argument(parser)
        self.__add_events_argument(parser)
        self.__add_all_measures_argument(parser)
        self.__add_tesla_argument(parser)
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
    
    def show_verbose(self) -> bool:
        """
        Check if program has to show verbose.

        Returns:
            True to verbose of False if not
        """

        return self.__verbose
        pass

    def show_desc(self) -> bool:
        """
        Check if program has to show description of results.

        Returns:
            True to show description of False if not
        """

        return self.__show_desc
        pass

    def show_metrics(self) -> bool:
        """
        Check if program has to show metrics computed by NVIDIA scan tool.

        Returns:
            True if program has to show metrics computed by NVIDIA scan tool or
            False if not.
        """

        return self.__show_metrics
        pass

    def show_all_measures(self) -> bool:
        """
        Check if program has to show all measures (events and metrics) computed by NVIDIA scan tool.

        Returns:
            True if program has to show all measures (events and metrics) computed by NVIDIA scan tool or
            False if not.
        """

        return self.__show_all_measurementes 
        pass

    def show_events(self) -> bool:
        """
        Check if program has to show events computed by NVIDIA scan tool.

        Returns:
            True if program has to show events computed by NVIDIA scan tool or
            False if not.
        """

        return self.__show_events
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
        message : str = "TopDown Metholodgy over NVIDIA's GPUs"
        #printer.print_center_msg_box(msg = "TopDown Metholodgy over NVIDIA's GPUs", indent = 1, title = "", output_file = self.output_file(), 
        #        width = None, delete_content_file = self.delete_output_file_content())
        printer.print_desplazed_underlined_str(message = message, output_file = self.output_file(), delete_content_file = False)
        print()
        print()
        print()
        message  = "\n\n\nWelcome to the " + sys.argv[0] + " program where you can check the bottlenecks of your\n" + \
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
                   "- Output File:                      " + str(self.output_file()) + "\n" + \
                   "- Verbose:                          " + str(self.show_verbose()) + "\n" + \
                   "- Delete output's file content:     " + str(self.delete_output_file_content()))
        printer.print_msg_box(msg = message, indent = 1, title = "EXECUTION FEATURES", output_file = self.output_file(), width = None,
            delete_content_file = False)
        print()
        message = "\n\nSaid that, according to the level entered by you, WE START THE ANALYSIS.\n"
        printer.print_max_line_length_message(message = message, max_length = TopDownParameters.C_NUM_MAX_CHARACTERS_PER_LINE, output_file = self.output_file(), delete_content_file = False)
        print()
        pass
      
    def __show_level_one_results(self, level_execution : LevelOne):
        stalls_front_message : str = ("{:<20} {:<6}".format('STALLS, on the total (%): ', 
            str(round(level_execution.get_front_end_stall(), TopDownParameters.C_MAX_NUM_RESULTS_DECIMALS)) + '%'))
        stalls_back_message : str = ("{:<20} {:<6}".format('STALLS, on the total (%): ', 
            str(round(level_execution.get_back_end_stall(), TopDownParameters.C_MAX_NUM_RESULTS_DECIMALS)) + '%'))
        ipc_degradation_divergence_message : str = ("{:<20} {:<6}".format("IPC DEGRADATION (%): ", 
            str(round(level_execution.divergence_percentage_ipc_degradation(), TopDownParameters.C_MAX_NUM_RESULTS_DECIMALS)) + '%'))   
        ipc_retire_message : str = ("{:<21} {:<4}".format('PERFORMANCE IPC (%):', 
            str(round(level_execution.retire_ipc_percentage(), TopDownParameters.C_MAX_NUM_RESULTS_DECIMALS)) + '%')) 

        ipc_degradation_front_message : str = ("{:<26} {:<5}".format('IPC DEGRADATION      (%): ', 
            str(round(level_execution.front_end_percentage_ipc_degradation(), TopDownParameters.C_MAX_NUM_RESULTS_DECIMALS)) + '%'))
        ipc_degradation_back_message : str = ("{:<26} {:<5}".format('IPC DEGRADATION      (%): ', 
            str(round(level_execution.back_end_percentage_ipc_degradation(), TopDownParameters.C_MAX_NUM_RESULTS_DECIMALS)) + '%'))
        
        messages : list[list[str]] = [[stalls_front_message, stalls_back_message, ipc_degradation_divergence_message, ipc_retire_message],
        [ipc_degradation_front_message, ipc_degradation_back_message, " ", " "]]

        titles : list[str] = [level_execution.front_end().name(), level_execution.back_end().name(),
            level_execution.divergence().name(),level_execution.retire().name()]

        MessageFormat().print_four_msg_box(messages, titles, 1, self.output_file(), self.delete_output_file_content())
        pass

    def __show_level_two_results(self, level_execution : LevelTwo):
        stalls_front_band_width_on_total_message : str = ("{:<20} {:<6}".format('STALLS, on the total (%): ', 
            str(round(level_execution.get_front_band_width_stall(), TopDownParameters.C_MAX_NUM_RESULTS_DECIMALS)) + '%'))
        stalls_front_dependency_on_total_message : str = ("{:<20} {:<6}".format('STALLS, on the total (%): ', 
            str(round(level_execution.get_front_dependency_stall(), TopDownParameters.C_MAX_NUM_RESULTS_DECIMALS)) + '%'))
        stalls_back_core_bound_on_total_message : str = ("{:<20} {:<6}".format('STALLS, on the total (%): ', 
            str(round(level_execution.get_back_core_bound_stall(), TopDownParameters.C_MAX_NUM_RESULTS_DECIMALS)) + '%'))
        stalls_back_memory_bound_on_total_message : str = ("{:<20} {:<6}".format('STALLS, on the total (%): ', 
            str(round(level_execution.get_back_memory_bound_stall(), TopDownParameters.C_MAX_NUM_RESULTS_DECIMALS)) + '%'))

        stalls_front_band_width_on_front_message : str = ("{:<22} {:<6}".format('STALLS, on FrontEnd  (%): ', 
            str(round(level_execution.get_front_band_width_stall_on_front(), TopDownParameters.C_MAX_NUM_RESULTS_DECIMALS)) + '%'))
        stalls_front_dependency_on_front_message : str = ("{:<20} {:<6}".format('STALLS, on FrontEnd  (%): ', 
            str(round(level_execution.get_front_dependency_stall_on_front(), TopDownParameters.C_MAX_NUM_RESULTS_DECIMALS)) + '%'))
        stalls_back_core_bound_on_back_message : str = ("{:<20} {:<6}".format('STALLS, on BackEnd   (%): ', 
            str(round(level_execution.get_back_core_bound_stall_on_back(), TopDownParameters.C_MAX_NUM_RESULTS_DECIMALS)) + '%'))
        stalls_back_memory_bound_on_back_message : str = ("{:<20} {:<6}".format('STALLS, on BackEnd   (%): ', 
            str(round(level_execution.get_back_memory_bound_stall_on_back(), TopDownParameters.C_MAX_NUM_RESULTS_DECIMALS)) + '%'))


        ipc_degradation_front_band_width_message : str = ("{:<26} {:<5}".format('IPC DEGRADATION      (%): ', 
            str(round(level_execution.front_band_width_percentage_ipc_degradation(), TopDownParameters.C_MAX_NUM_RESULTS_DECIMALS)) + '%'))
        ipc_degradation_front_dependency_message : str = ("{:<26} {:<5}".format('IPC DEGRADATION      (%): ', 
            str(round(level_execution.front_dependency_percentage_ipc_degradation(), TopDownParameters.C_MAX_NUM_RESULTS_DECIMALS)) + '%'))
        ipc_degradation_back_core_bound_message : str = ("{:<26} {:<5}".format('IPC DEGRADATION      (%): ', 
            str(round(level_execution.back_core_bound_percentage_ipc_degradation(), TopDownParameters.C_MAX_NUM_RESULTS_DECIMALS)) + '%'))
        ipc_degradation_back_memory_bound_message : str = ("{:<26} {:<5}".format('IPC DEGRADATION      (%): ', 
            str(round(level_execution.back_memory_bound_percentage_ipc_degradation(), TopDownParameters.C_MAX_NUM_RESULTS_DECIMALS)) + '%'))
       
        messages : list[list[str]] = [[stalls_front_band_width_on_total_message, stalls_front_dependency_on_total_message, 
            stalls_back_core_bound_on_total_message,stalls_back_memory_bound_on_total_message],
            [stalls_front_band_width_on_front_message, stalls_front_dependency_on_front_message, 
            stalls_back_core_bound_on_back_message, stalls_back_memory_bound_on_back_message],
            ["", "", "", ""], [ipc_degradation_front_band_width_message, ipc_degradation_front_dependency_message,
            ipc_degradation_back_core_bound_message, ipc_degradation_back_memory_bound_message]]

        titles : list[str] = [level_execution.front_band_width().name(), level_execution.front_dependency().name(),
            level_execution.back_core_bound().name(),level_execution.back_memory_bound().name()]

        MessageFormat().print_four_msg_box(messages, titles, 1, self.output_file(), self.delete_output_file_content())
        pass

    def __show_level_three_results(self, level_execution : LevelThree):
        stalls_constant_memory_bound_on_total_message : str = ("{:<20} {:<6}".format('STALLS, on the total            (%): ', # revisar formatos, quiza sobren TODO
            str(round(level_execution.get_constant_memory_bound_stall(), TopDownParameters.C_MAX_NUM_RESULTS_DECIMALS)) + '%'))
        
        stalls_constant_memory_bound_on_memory_bound_message : str = ("{:<22} {:<6}".format('STALLS, on BackEnd.MemoryBound  (%): ', 
            str(round(level_execution.get_constant_memory_bound_stall_on_memory_bound(), TopDownParameters.C_MAX_NUM_RESULTS_DECIMALS)) + '%'))
       
        stalls_constant_memory_bound_on_back_message : str = ("{:<22} {:<6}".format('STALLS, on MemoryBound          (%): ', 
            str(round(level_execution.get_constant_memory_bound_stall_on_back(), TopDownParameters.C_MAX_NUM_RESULTS_DECIMALS)) + '%'))

        ipc_degradation_constant_memory_bound_width_message : str = ("{:<26} {:<5}".format('IPC DEGRADATION                 (%): ', 
            str(round(level_execution.constant_memory_bound_percentage_ipc_degradation(), TopDownParameters.C_MAX_NUM_RESULTS_DECIMALS)) + '%'))
       
        messages : list[list[str]] = [[stalls_constant_memory_bound_on_total_message, "", "", ""],
                [stalls_constant_memory_bound_on_memory_bound_message, "", "", ""],
                [stalls_constant_memory_bound_on_back_message, "", "", ""],
                ["","","",""], [ipc_degradation_constant_memory_bound_width_message, "", "", ""]]

        titles : list[str] = [level_execution.constant_memory_bound().name(), "", "", ""]

        MessageFormat().print_four_msg_box(messages, titles, 1, self.output_file(), self.delete_output_file_content())
        pass
    
    def __show_results(self, level_execution):
        """ Show Results of execution indicated by argument.

        Parameters:
            level_one   : LevelOne/LevelTwo/LevelThree  ; reference to level one/two/three analysis ALREADY DONE.
        """

        printer : MessageFormat = MessageFormat()
        message : str = "The results have been obtained correctly. General results of IPC are the following:\n\n"
        printer.print_max_line_length_message(message = message, max_length = TopDownParameters.C_NUM_MAX_CHARACTERS_PER_LINE, output_file = self.output_file(), 
        delete_content_file = False)
        print()
        message = ("IPC OBTAINED: " + str(round(level_execution.ipc(),TopDownParameters.C_MAX_NUM_RESULTS_DECIMALS)) + " | MAXIMUM POSSIBLE IPC: " +  
            str(level_execution.get_device_max_ipc()))
        printer.print_desplazed_msg_box(msg = message, indent = 1, title = "", output_file = self.output_file(), width = None, delete_content_file = False)
        if self.show_desc():
            message = ("\n\n'IPC OBTAINED' is the IPC of the analyzed program (computed by scan tool) and 'MAXIMUM POSSIBLE IPC'\n" +
                "is the the maximum IPC your GPU can achieve. This is computed taking into account architectural concepts, such as the\n" +
                "number of warp planners per SM, as well as the number of Dispatch units of each SM.")
            printer.print_max_line_length_message(message = message, max_length = TopDownParameters.C_NUM_MAX_CHARACTERS_PER_LINE, output_file = self.output_file(), 
            delete_content_file = False)
            message = ("\n    As you can see, the IPC obtanied it " + "is " + str(round((level_execution.get_device_max_ipc()/level_execution.ipc())*100, 
                TopDownParameters.C_MAX_NUM_RESULTS_DECIMALS)) + "% smaller than you could get. This lower IPC is due to STALLS in the different \nparts " +
                "of the architecture and DIVERGENCE problems. " +
                "We analyze them based on the level of the TopDown:\n")
            printer.print_max_line_length_message(message = message, max_length = TopDownParameters.C_NUM_MAX_CHARACTERS_PER_LINE, output_file = self.output_file(), 
            delete_content_file = False)
            print()
        message = "DESCRIPTION OF MEASURE PARTS"
        printer.print_desplazed_underlined_str(message = message, output_file = self.output_file(), delete_content_file = False)
        print()
        if type(level_execution) is LevelTwo or type(level_execution) is LevelThree:
            message = "LEVEL ONE RESULTS"
            printer.print_underlined_str(message = message, output_file = self.output_file(), delete_content_file = False)
            print()
            if self.show_desc():
                message = "\n" + level_execution.front_end().name() + ": " + level_execution.front_end().description() + "\n\n"
                printer.print_max_line_length_message(message = message, max_length = TopDownParameters.C_NUM_MAX_CHARACTERS_PER_LINE, output_file = self.output_file(), 
                delete_content_file = False)
                print()
                message = "\n" + level_execution.back_end().name() + ": " + level_execution.back_end().description() + "\n\n"
                printer.print_max_line_length_message(message = message, max_length = TopDownParameters.C_NUM_MAX_CHARACTERS_PER_LINE, output_file = self.output_file(), 
                delete_content_file = False)
                print()
                message = "\n" + level_execution.divergence().name() + ": " + level_execution.divergence().description() + "\n\n"
                printer.print_max_line_length_message(message = message, max_length = TopDownParameters.C_NUM_MAX_CHARACTERS_PER_LINE, 
                output_file = self.output_file(), delete_content_file = False)
                print()
            self.__show_level_one_results(level_execution)
            print()
            message = "LEVEL TWO RESULTS"
            printer.print_underlined_str(message = message, output_file = self.output_file(), delete_content_file = False)
            print()
            if self.show_desc():
                message = "\n" + level_execution.front_band_width().name() + ": " + level_execution.front_band_width().description() + "\n\n"
                printer.print_max_line_length_message(message = message, max_length = TopDownParameters.C_NUM_MAX_CHARACTERS_PER_LINE, 
                output_file = self.output_file(), delete_content_file = False)
                print()
                message = "\n" + level_execution.front_dependency().name() + ": " + level_execution.front_dependency().description() + "\n\n"
                printer.print_max_line_length_message(message = message, max_length = TopDownParameters.C_NUM_MAX_CHARACTERS_PER_LINE, 
                output_file = self.output_file(), delete_content_file = False)
                print()
                message = "\n" + level_execution.back_core_bound().name() + ": " + level_execution.back_core_bound().description() + "\n\n"
                printer.print_max_line_length_message(message = message, max_length = TopDownParameters.C_NUM_MAX_CHARACTERS_PER_LINE, 
                output_file = self.output_file(), delete_content_file = False)
                print()
                message = "\n" + level_execution.back_memory_bound().name() + ": " + level_execution.back_memory_bound().description() + "\n\n"
                printer.print_max_line_length_message(message = message, max_length = TopDownParameters.C_NUM_MAX_CHARACTERS_PER_LINE, 
                output_file = self.output_file(), delete_content_file = False)
                print()
            self.__show_level_two_results(level_execution)
            print()
            if type(level_execution) is LevelThree:
                message = "LEVEL THREE RESULTS"
                printer.print_underlined_str(message = message, output_file = self.output_file(), delete_content_file = False)
                print()
                if self.show_desc():
                    message = "\n" + level_execution.constant_memory_bound().name() + ": " + level_execution.constant_memory_bound().description() + "\n\n"
                    printer.print_max_line_length_message(message = message, max_length = TopDownParameters.C_NUM_MAX_CHARACTERS_PER_LINE, 
                    output_file = self.output_file(), delete_content_file = False)
                    print()
                self.__show_level_three_results(level_execution)
                print()
        else: # level one
            message = "RESULTS"
            printer.print_underlined_str(message = message, output_file = self.output_file(), delete_content_file = False)
            print()
            if self.show_desc():
                message = "\n" + level_execution.front_end().name() + ": " + level_execution.front_end().description() + "\n\n"
                printer.print_max_line_length_message(message = message, max_length = TopDownParameters.C_NUM_MAX_CHARACTERS_PER_LINE, output_file = self.output_file(), delete_content_file = False)
                print()
                message = "\n" + level_execution.back_end().name() + ": " + level_execution.back_end().description() + "\n\n"
                printer.print_max_line_length_message(message = message, max_length = TopDownParameters.C_NUM_MAX_CHARACTERS_PER_LINE, output_file = self.output_file(), 
                delete_content_file = False)
                print()
                message = "\n" + level_execution.divergence().name() + ": " + level_execution.divergence().description() + "\n\n"
                printer.print_max_line_length_message(message = message, max_length = TopDownParameters.C_NUM_MAX_CHARACTERS_PER_LINE, 
                output_file = self.output_file(), delete_content_file = False)
                print()
            self.__show_level_one_results(level_execution)
            print()
        pass

    def launch(self):
        """ Launch execution."""
        
        if self.show_verbose():
            # introduction
            self.__intro_message()

        show_metrics : bool = self.show_metrics()
        show_events : bool = self.show_events()
        if self.show_all_measures():
            show_metrics = True
            show_events = True
        if self.level() == 1:
           level : LevelOne = LevelOne(self.program(), self.output_file(), show_metrics, show_events)
        elif self.level() == 2:
            level : LevelTwo = LevelTwo(self.program(), self.output_file(), show_metrics, show_events) 
        elif self.level() == 3:
             level : LevelThree = LevelThree(self.program(), self.output_file(), show_metrics, show_events) 
        lst_output : list[str] = list() # for extra information
        level.run(lst_output)
        self.__show_results(level)
        if self.show_desc(): ### revisar este IF no deberia ir aqui
            # Write results in output-file if has been specified

            printer : MessageFormat = MessageFormat()
            print()
            message : str = "List of measurements computed by NVIDIA scan tool"
            printer.print_desplazed_underlined_str(message = message, output_file = self.output_file(), delete_content_file = False)
            print()
            print()
        if self.show_all_measures() or self.show_metrics() or self.show_events():
            if not self.output_file() is None:
                    printer.write_in_file_at_end(self.output_file(), lst_output)
            element : str
            for element in lst_output:
                print(element)
    
if __name__ == '__main__':
    td = TopDown()
    td.launch()
    MessageFormat().print_max_line_length_message(message = "Analysis performed correctly!", 
    max_length = TopDownParameters.C_NUM_MAX_CHARACTERS_PER_LINE, output_file = td.output_file(), delete_content_file = False)