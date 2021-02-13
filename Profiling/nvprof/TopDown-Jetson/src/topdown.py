"""
Program that implements the Top Down methodology over NVIDIA GPUs.

@author:    Alvaro Saiz (UC)
@date:      Jan 2021
@version:   1.0
"""

import argparse
import re
#from tabulate import tabulate #TODO, pintar
from errors import *    
from shell import Shell # launch shell arguments
from params import Parameters # parameters of program

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
    pass
    
    def __add_program_argument(self, group : argparse._ArgumentGroup) :
        """ 
        Add program argument. 'C_FILE_SHORT_OPTION' is the short option of argument
        and 'C_FILE_LONG_OPTION' is the long version of argument.

        Params:
            group : argparse._ArgumentGroup ; group of the argument.
        """

        group.add_argument(
            Parameters.C_FILE_SHORT_OPTION, 
            Parameters.C_FILE_LONG_OPTION, 
            required = True,
            help = 'run file. Path to file.',
            default = None,
            nargs = '?', 
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
            Parameters.C_LEVEL_SHORT_OPTION, Parameters.C_LEVEL_LONG_OPTION,
            required = True,
            help = 'level of execution.',
            type = int,
            nargs = 1,
            default = -1,
            choices = range(Parameters.C_MIN_LEVEL_EXECUTION, Parameters.C_MAX_LEVEL_EXECUTION + 1), # range [1,2], produces error, no if needed
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
            Parameters.C_OUTPUT_FILE_SHORT_OPTION, 
            Parameters.C_OUTPUT_FILE_LONG_OPTION, 
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
            Parameters.C_LONG_DESCRIPTION_SHORT_OPTION, 
            Parameters.C_LONG_DESCRIPTION_LONG_OPTION, 
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

    def __write_in_file_at_end(self, file : str, message : list[str]):
        """
        Write 'message' at the end of file with path 'file'

        Params:
            file    : str           ; path to file to write.
            message : list[str]     ; list of string with the information to be written (in order) to file.
                                      Each element of the list corresponds to a line.

        Raises:
            WriteInOutPutFileError  ; error when opening or write in file. Operation not performed
        """

        try:
            f : _io.TextIOWrapper = open(file, "a")
            try:
                f.write("\n".join(str(item) for item in message))

            finally:
                f.close()
        except:  
            raise WriteInOutPutFileError
    pass

    def __add_result_part_to_lst(self, dict_values : dict, dict_desc : dict, message : str, 
        lst_to_add : list[str], isMetric : bool):
        """
        Add results of execution part (FrontEnd, BackEnd...) to list indicated by argument.

        Params:
            dict_values     : dict      ; diccionary with name_metric/event-value elements of the part to 
                                          add to 'lst_to_add'
            dict_desc       : dict      ; diccionary with name_metric/event-description elements of the 
                                          part to add to 'lst_to_add'
            message         : str       ; introductory message to append to 'lst_to_add' to delimit 
                                          the beginning of the region
            lst_output      : list[str] ; list where to add all elements
            isMetric        : bool      ; True if they are metrics or False if they are events

        Raises:
            MetricNoDefined             ; raised in case you have added an metric that is 
                                          not supported or does not exist in the NVIDIA analysis tool
            EventNoDefined              ; raised in case you have added an event that is 
                                          not supported or does not exist in the NVIDIA analysis tool
        """
        
        lst_to_add.append(message)
        lst_to_add.append( "\t\t\t----------------------------------------------------"
            + "---------------------------------------------------")
        if isMetric:
            lst_to_add.append("\t\t\t{:<45} {:<48}  {:<5} ".format('Metric Name','Metric Description', 'Value'))
            lst_to_add.append( "\t\t\t----------------------------------------------------"
            +"---------------------------------------------------")
            for (metric_name, value), (metric_name, desc) in zip(dict_values.items(), 
                dict_desc.items()):
                if metric_name is None or desc is None or value is None:
                    print(str(desc) + str(value) + str(isMetric))
                    raise MetricNoDefined(metric_name)
                lst_to_add.append("\t\t\t{:<45} {:<49} {:<6} ".format(metric_name, desc, value))
        else:
            lst_to_add.append("\t\t\t{:<45} {:<46}  {:<5} ".format('Event Name','Event Description', 'Value'))
            lst_to_add.append( "\t\t\t----------------------------------------------------"
            +"---------------------------------------------------")
            #for (counter, value), (counter, desc) in zip(dict_values.items(), 
            #    dict_desc.items()):
            #    if counter is None or desc is None or value is None:
            #        print(str(desc) + str(value) + str(isMetric))
            #        raise MetricNoDefined(counter)
            #    lst_to_add.append("\t\t\t{:<45} {:<50} {:<6} ".format(counter, desc, value))
            value_event : str 
            for event_name in dict_values:
                value_event = dict_values.get(event_name)
                if event_name is None or value_event is None:
                    print(str(event_name) + " " + str(value_event))
                    raise EventNoDefined(event_name)
                lst_to_add.append("\t\t\t{:<45} {:<47} {:<6} ".format(event_name, "-", value_event))
        lst_to_add.append("\t\t\t----------------------------------------------------"
            +"---------------------------------------------------")
    pass

    def level_1(self):
        """ 
        Run TopDown level 1.

        Raises:
            EventNotAsignedToPart       ; raised when an event has not been assigned to any analysis part 
            MetricNotAsignedToPart      ; raised when a metric has not been assigned to any analysis part
            ProfilingError              ; raised in case of error reading results from NVIDIA scan tool
        """

        shell : Shell = Shell()
        # Command to launch
        command : str = ("sudo $(which nvprof) --metrics " + Parameters.C_LEVEL_1_FRONT_END_METRICS + 
            "," + Parameters.C_LEVEL_1_BACK_END_METRICS + "," + Parameters.C_LEVEL_1_DIVERGENCE_METRICS 
            + "  --events " + Parameters.C_LEVEL_1_FRONT_END_EVENTS + 
            "," + Parameters.C_LEVEL_1_BACK_END_EVENTS + "," + Parameters.C_LEVEL_1_DIVERGENCE_EVENTS + 
            " --unified-memory-profiling off --profile-from-start off " + self.__program)
        output_file : str = self.output_file()
        output_command : bool
        if output_file is None:
            output_command = shell.launch_command(command, Parameters.C_INFO_MESSAGE_EXECUTION_NVPROF)
        else:
            output_command = shell.launch_command_redirect(command, Parameters.C_INFO_MESSAGE_EXECUTION_NVPROF, output_file, True)
        if output_command is None:
            raise ProfilingError
        else:
            dict_front_metrics : dict = dict() 
            dict_front_metrics_desc : dict = dict()
            dict_back_metrics : dict = dict()
            dict_back_metrics_desc : dict = dict()
            dict_divergence_metrics : dict = dict()
            dict_divergence_metrics_desc : dict = dict() 

            dict_front_events : dict = dict() 
            dict_front_events_desc : dict = dict() 
            dict_back_events : dict = dict() 
            dict_back_events_desc : dict = dict() 
            dict_divergence_events : dict = dict() 
            dict_divergence_events_desc : dict = dict() 

            # Create dictionaries with name of counters as key.
            if Parameters.C_LEVEL_1_FRONT_END_METRICS != "":
                dict_front_metrics = dict.fromkeys(Parameters.C_LEVEL_1_FRONT_END_METRICS.split(","))
                dict_front_metrics_desc = dict.fromkeys(Parameters.C_LEVEL_1_FRONT_END_METRICS.split(","))
            if Parameters.C_LEVEL_1_BACK_END_METRICS != "":
                dict_back_metrics = dict.fromkeys(Parameters.C_LEVEL_1_BACK_END_METRICS.split(","))
                dict_back_metrics_desc = dict.fromkeys(Parameters.C_LEVEL_1_BACK_END_METRICS.split(","))
            if Parameters.C_LEVEL_1_DIVERGENCE_METRICS != "":
                dict_divergence_metrics = dict.fromkeys(Parameters.C_LEVEL_1_DIVERGENCE_METRICS.split(","))
                dict_divergence_metrics_desc = dict.fromkeys(Parameters.C_LEVEL_1_DIVERGENCE_METRICS.split(","))   
            if Parameters.C_LEVEL_1_FRONT_END_EVENTS != "":
                dict_front_events = dict.fromkeys(Parameters.C_LEVEL_1_FRONT_END_EVENTS.split(","))
                dict_front_events_desc = dict.fromkeys(Parameters.C_LEVEL_1_FRONT_END_EVENTS.split(","))
            if Parameters.C_LEVEL_1_BACK_END_EVENTS != "":
                dict_back_events = dict.fromkeys(Parameters.C_LEVEL_1_BACK_END_EVENTS.split(","))
                dict_back_events_desc = dict.fromkeys(Parameters.C_LEVEL_1_BACK_END_EVENTS.split(","))
            if Parameters.C_LEVEL_1_DIVERGENCE_EVENTS!= "":
                dict_divergence_events = dict.fromkeys(Parameters.C_LEVEL_1_DIVERGENCE_EVENTS.split(","))
                dict_divergence_events_desc = dict.fromkeys(Parameters.C_LEVEL_1_DIVERGENCE_EVENTS.split(","))

            line : str
            list_words : list[str]
            
            # events
            event_name : str
            event_total_value : str

            #metrics 
            metric_name : str
            metric_description : str = ""
            metric_avg_value : str 
            #metric_max_value : str 
            #metric_min_value : str

            #control
            has_read_all_events : bool = False
            line : str
            i : int
            list_words : list[str]
            has_found_part : bool = False
            for line in output_command.splitlines():
                line = re.sub(' +', ' ', line) # delete more than one spaces and put only one
                list_words = line.split(" ")
                has_found_part = False
                if not has_read_all_events:
                    # Check if it's line of interest:
                    # ['', 'X', 'event_name','Min', 'Max', 'Avg', 'Total'] event_name is str. Rest: numbers (less '', it's an space)
                    if len(list_words) > 1: 
                        if list_words[1] == "Metric": # check end events
                            has_read_all_events = True
                        elif list_words[0] == '' and list_words[len(list_words) - 1][0].isnumeric():
                            event_name = list_words[2]
                            event_total_value = list_words[len(list_words) - 1]
                            if (Parameters.C_LEVEL_1_FRONT_END_EVENTS != "" and event_name in dict_front_events):
                                dict_front_events[event_name] = event_total_value
                                #dict_front_events_desc[name_counter] = description_counter
                                has_found_part = True
                            if (Parameters.C_LEVEL_1_BACK_END_EVENTS != "" and event_name in dict_back_events):
                                dict_back_events[event_name] = event_total_value
                                #dict_back_events_desc[name_counter] = description_counter
                                has_found_part = True
                            if (Parameters.C_LEVEL_1_DIVERGENCE_EVENTS != "" and event_name in dict_divergence_events): 
                                dict_divergence_events[event_name] = event_total_value
                                #dict_divergence_events_desc[name_counter] = description_counter
                                has_found_part = True
                            if not has_found_part:
                                raise EventNotAsignedToPart(event_name)     
                else: # metrics
                    # Check if it's line of interest:
                    # ['', 'X', 'NAME_COUNTER', ... , 'Min', 'Max', 'Avg' (Y%)] where X (int number), Y (int/float number)
                    if len(list_words) > 1 and list_words[0] == '' and list_words[len(list_words) - 1][0].isnumeric():
                        metric_name = list_words[2]
                        metric_description = ""
                        for i in range(3, len(list_words) - 3):
                            metric_description += list_words[i] + " "     
                        metric_avg_value = list_words[len(list_words) - 1]
                        #metric_max_value = list_words[len(list_words) - 2]
                        #metric_min_value = list_words[len(list_words) - 3]

                        #if metric_avg_value != metric_max_value or metric_avg_value != metric_min_value:
                            # Do Something. NOT USED
                        if (Parameters.C_LEVEL_1_FRONT_END_METRICS != "" and metric_name in # metrics
                            (dict_front_metrics and dict_front_metrics_desc)):
                            dict_front_metrics[metric_name] = metric_avg_value
                            dict_front_metrics_desc[metric_name] = metric_description
                            has_found_part = True
                        if (Parameters.C_LEVEL_1_BACK_END_METRICS != "" and metric_name in 
                            (dict_back_metrics and dict_back_metrics_desc)):
                            dict_back_metrics[metric_name] = metric_avg_value
                            dict_back_metrics_desc[metric_name] = metric_description
                            has_found_part = True 
                        if (Parameters.C_LEVEL_1_DIVERGENCE_METRICS != "" and metric_name in 
                            (dict_divergence_metrics and dict_divergence_metrics_desc)): 
                            dict_divergence_metrics[metric_name] = metric_avg_value
                            dict_divergence_metrics_desc[metric_name] = metric_description
                            has_found_part = True
                        if not has_found_part:
                            raise MetricNotAsignedToPart(metric_name)
            #  Keep Results
            lst_output : list[str] = list()
            lst_output.append("\nList of counters/metrics measured according to the part.")
            if Parameters.C_LEVEL_1_FRONT_END_METRICS != "":
                self.__add_result_part_to_lst(dict_front_metrics, 
                    dict_front_metrics_desc,"\n- FRONT-END RESULTS:", lst_output, True)
            if Parameters.C_LEVEL_1_FRONT_END_EVENTS != "":
                    self.__add_result_part_to_lst(dict_front_events, 
                    dict_front_events_desc,"", lst_output, False)
            if Parameters.C_LEVEL_1_BACK_END_METRICS != "":
                self.__add_result_part_to_lst(dict_back_metrics, 
                    dict_back_metrics_desc,"\n- BACK-END RESULTS:", lst_output, True)
            if Parameters.C_LEVEL_1_BACK_END_EVENTS != "":
                    self.__add_result_part_to_lst(dict_back_events, 
                    dict_back_events_desc,"", lst_output, False)
            if Parameters.C_LEVEL_1_DIVERGENCE_METRICS != "":
                self.__add_result_part_to_lst(dict_divergence_metrics, 
                    dict_divergence_metrics_desc,"\n\n- DIVERGENCE RESULTS:", lst_output, True)
            if Parameters.C_LEVEL_1_DIVERGENCE_EVENTS != "":
                self.__add_result_part_to_lst(dict_divergence_events, 
                    dict_divergence_events_desc, "", lst_output, False)
            lst_output.append("\n")
            if self.show_long_desc():
                # Write results in output-file if has been specified
                if not self.output_file() is None:
                    self.__write_in_file_at_end(self.output_file(), lst_output)
                element : str
                for element in lst_output:
                    print(element)
    pass

    def level_2(self):
        """ 
        Run TopDown level 2.
        """
        # TODO
    pass
    
if __name__ == '__main__':
    td = TopDown()
    level : int = td.level()

    if level == 1:
        td.level_1()
    elif level == 2:
        td.level_2()
    print("Analysis performed correctly!")