"""
Class that represents the level one of the execution

@author:    Alvaro Saiz (UC)
@date:      Jan-2021
@version:   1.0
"""
import re
import sys
sys.path.insert(1, '/home/alvaro/Documents/Facultad/TopDownNvidia/Profiling/nvprof/TopDown-Jetson/src/errors')
sys.path.insert(1, '/home/alvaro/Documents/Facultad/TopDownNvidia/Profiling/nvprof/TopDown-Jetson/src/parameters')
sys.path.insert(1, '/home/alvaro/Documents/Facultad/TopDownNvidia/Profiling/nvprof/TopDown-Jetson/src/measure_parts')
sys.path.insert(1, '/home/alvaro/Documents/Facultad/TopDownNvidia/Profiling/nvprof/TopDown-Jetson/src/measure_levels')
from level_execution import LevelExecution
from level_execution_params import LevelExecutionParameters
from shell.shell import Shell # launch shell arguments
from level_execution_errors import *

class LevelOne(LevelExecution):
    """ 
    Class thath represents the level one of the execution.
    """
    
    def __get_results_level_1(self, lst_output : list[str]):
        """ 
        Run TopDown level 1 and get results of the different parts of the level 1.

        Parameters:
            lst_output      : list[str]     ; OUTPUT list with results
                
        Raises:
            EventNotAsignedToPart       ; raised when an event has not been assigned to any analysis part 
            MetricNotAsignedToPart      ; raised when a metric has not been assigned to any analysis part
            ProfilingError              ; raised in case of error reading results from NVIDIA scan tool
        """

        shell : Shell = Shell()
        # Command to launch
        command : str = ("sudo $(which nvprof) --metrics " + self._front_end.metrics_str() + 
            "," + self._back_end.metrics_str() + "," + self._divergence.metrics_str() + "," + self._extra_measure.metrics_str()
            + "," + self._retire.metrics_str() + "  --events " + self._front_end.events_str() + 
            "," + self._back_end.events_str() + "," + self._divergence.events_str() +  "," + self._extra_measure.events_str() +
             "," + self._retire.events_str() + " --unified-memory-profiling off --profile-from-start off " + self._program)
        output_file : str = self._output_file
        output_command : bool
        if output_file is None:
            output_command = shell.launch_command(command, LevelExecutionParameters.C_INFO_MESSAGE_EXECUTION_NVPROF)
        else:
            output_command = shell.launch_command_redirect(command, LevelExecutionParameters.C_INFO_MESSAGE_EXECUTION_NVPROF, output_file, True)
        if output_command is None:
            raise ProfilingError
        else:
            event_name : str
            event_total_value : str 
            metric_name : str
            metric_description : str = ""
            metric_avg_value : str 
            #metric_max_value : str 
            #metric_min_value : str
            has_read_all_events : bool = False
            line : str
            i : int
            list_words : list[str]
            front_end_value_has_found : bool
            frond_end_description_has_found : bool
            back_end_value_has_found : bool
            back_end_description_has_found : bool
            divergence_value_has_found : bool
            divergence_description_has_found : bool
            extra_measure_value_has_found : bool
            extra_measure_description_has_found : bool
            retire_value_has_found : bool 
            retire_description_has_found : bool
            for line in output_command.splitlines():
                line = re.sub(' +', ' ', line) # delete more than one spaces and put only one
                list_words = line.split(" ")
                if not has_read_all_events:
                    # Check if it's line of interest:
                    # ['', 'X', 'event_name','Min', 'Max', 'Avg', 'Total'] event_name is str. Rest: numbers (less '', it's an space)
                    if len(list_words) > 1: 
                        if list_words[1] == "Metric": # check end events
                            has_read_all_events = True
                        elif list_words[0] == '' and list_words[len(list_words) - 1][0].isnumeric():
                            event_name = list_words[2]
                            event_total_value = list_words[len(list_words) - 1]     
                            front_end_value_has_found = self._front_end.set_event_value(event_name, event_total_value)
                            #frond_end_description_has_found = front_end.set_event_description(event_name, metric_description)
                            back_end_value_has_found = self._back_end.set_event_value(event_name, event_total_value)
                            #back_end_description_has_found = back_end.set_event_description(event_name, metric_description)
                            divergence_value_has_found = self._divergence.set_event_value(event_name, event_total_value)
                            #divergence_description_has_found = divergence.set_event_description(event_name, metric_description)
                            extra_measure_value_has_found = self._extra_measure.set_event_value(event_name, event_total_value)
                            #extra_measure_description_has_found = extra_measure.set_event_description(event_name, metric_description)
                            retire_value_has_found = self._retire.set_event_value(event_name, event_total_value)
                            #retire_description_has_found = extra_measure.set_event_description(event_name, metric_description)
                            if (not (front_end_value_has_found or back_end_value_has_found or divergence_value_has_found or 
                                extra_measure_value_has_found or retire_value_has_found)): #or 
                                #not(frond_end_description_has_found or back_end_description_has_found 
                                #or divergence_description_has_found or extra_measure_description_has_found)):
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
                        front_end_value_has_found = self._front_end.set_metric_value(metric_name, metric_avg_value)
                        frond_end_description_has_found = self._front_end.set_metric_description(metric_name, metric_description)
                        back_end_value_has_found = self._back_end.set_metric_value(metric_name, metric_avg_value)
                        back_end_description_has_found = self._back_end.set_metric_description(metric_name, metric_description)
                        divergence_value_has_found = self._divergence.set_metric_value(metric_name, metric_avg_value)
                        divergence_description_has_found = self._divergence.set_metric_description(metric_name, metric_description)
                        extra_measure_value_has_found = self._extra_measure.set_metric_value(metric_name, metric_avg_value)
                        extra_measure_description_has_found = self._extra_measure.set_metric_description(metric_name, metric_description)
                        retire_value_has_found = self._retire.set_metric_value(metric_name, metric_avg_value)
                        retire_description_has_found = self._retire.set_metric_description(metric_name, metric_description)
                        if (not (front_end_value_has_found or back_end_value_has_found or divergence_value_has_found or 
                            extra_measure_value_has_found or retire_value_has_found) or 
                            
                            not(frond_end_description_has_found or back_end_description_has_found or divergence_description_has_found or extra_measure_description_has_found 
                            or retire_description_has_found)):
                            raise MetricNotAsignedToPart(metric_name)
            #  Keep Results
            lst_output.append("\nList of counters/metrics measured according to the part.")
            if self._front_end.metrics_str() != "":
                self._add_result_part_to_lst(self._front_end.metrics(), 
                    self._front_end.metrics_description(),"\n- FRONT-END RESULTS:", lst_output, True)
            if self._front_end.events_str() != "":
                    self._add_result_part_to_lst(self._front_end.events(), 
                    self._front_end.events_description(), "", lst_output, False)
            if self._back_end.metrics_str() != "":
                self._add_result_part_to_lst(self._back_end.metrics(), 
                    self._back_end.metrics_description(),"\n- BACK-END RESULTS:", lst_output, True)
            if self._back_end.events_str() != "":
                    self._add_result_part_to_lst(self._back_end.events(), 
                    self._back_end.events_description(), "", lst_output, False)
            if self._divergence.metrics_str() != "":
                self._add_result_part_to_lst(self._divergence.metrics(), 
                    self._divergence.metrics_description(),"\n- DIVERGENCE RESULTS:", lst_output, True)
            if self._divergence.events_str() != "":
                    self._add_result_part_to_lst(self._divergence.events(), 
                    self._divergence.events_description(), "", lst_output, False)
            if self._retire.metrics_str() != "":
                    self._add_result_part_to_lst(self._retire.metrics(), 
                    self._retire.metrics_description(),"\n- RETIRE RESULTS:", lst_output, True)
            if self._retire.events_str() != "":
                    self._add_result_part_to_lst(self._retire.events(), 
                    self._retire.events_description(), "", lst_output, False)
            if self._extra_measure.metrics_str() != "":
                self._add_result_part_to_lst(self._extra_measure.metrics(), 
                    self._extra_measure.metrics_description(),"\n- EXTRA-MEASURE RESULTS:", lst_output, True)
            if self._extra_measure.events_str() != "":
                    self._add_result_part_to_lst(self._extra_measure.events(), 
                    self._extra_measure.events_description(), "", lst_output, False)
            lst_output.append("\n")
        pass

    def run(self, lst_output : list[str]):
        """Run execution."""
        
        self.__get_results_level_1(lst_output)
        pass
    