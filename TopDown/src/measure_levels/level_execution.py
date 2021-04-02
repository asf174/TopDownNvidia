"""
Class that represents the levels of the execution.

@author:    Alvaro Saiz (UC)
@date:      Jan-2021
@version:   1.0
"""

import locale
from abc import ABC, abstractmethod # abstract class
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from measure_parts.extra_measure import ExtraMeasure    
from shell.shell import Shell # launch shell arguments
from parameters.level_execution_params import LevelExecutionParameters # parameters of program
from errors.level_execution_errors import *
from parameters.topdown_params import TopDownParameters 

class LevelExecution(ABC):
    """ 
    Class that represents the levels of the execution.
     
    Attributes:
        _output_file            : str           ; path to output file with results. 'None' to don't use
                                                  output file
        _program                : str           ; program of the execution
        _recolect_metrics       : bool          ; True if the execution must recolted the metrics used by NVIDIA scan tool
                                                  or False in other case
        __compute_capability    : float         ; Compute Capbility of the execution
    """

    def __init__(self, program : str, output_file : str, recoltect_metrics : bool):
        self._program : str = program
        self._output_file : str = output_file
        self._recolect_metrics : bool = recoltect_metrics
        
        shell : Shell = Shell()
        compute_capability_str : str = shell.launch_command_show_all("nvcc ../src/measure_parts/compute_capability.cu --run", None)
        shell.launch_command("rm -f ../src/measure_parts/a.out", None)
        self.__compute_capability : float = float(compute_capability_str)
        if self.__compute_capability > TopDownParameters.C_COMPUTE_CAPABILITY_NVPROF_MAX_VALUE:
            locale.setlocale(locale.LC_ALL, 'es_ES.utf8')
        pass 

    @abstractmethod
    def _generate_command(self) -> str:
        """ 
        Generate command of execution with NVIDIA scan tool.

        Returns:
            String with command to be executed
        """

        pass

    @abstractmethod
    def run(self, lst_output): #: list):
        """
        Makes execution.
        
        Parameters:
            lst_output  : list ; list with results
        """
        
        pass

    def _launch(self, command : str) -> str:
        """ 
        Launch NVIDIA scan tool.
        
        Params:
            command : str ; String with command

        Returns:
            String with results.

        Raises:
            ProfilingError              ; raised in case of error reading results from NVIDIA scan tool
        """

        shell : Shell = Shell()
        #output_file : str = self._output_file
        output_command : bool
        # if que muestra el resultado del NVPROF en el fichero
        #if output_file is None:
        #    output_command = shell.launch_command(command, LevelExecutionParameters.C_INFO_MESSAGE_EXECUTION_NVPROF)
        #else:
        #    output_command = shell.launch_command_redirect(command, LevelExecutionParameters.C_INFO_MESSAGE_EXECUTION_NVPROF, 
        #        output_file, True)
        output_command = shell.launch_command(command, LevelExecutionParameters.C_INFO_MESSAGE_EXECUTION_NVPROF)
        if output_command is None:
            raise ProfilingError
        return output_command  
        pass
    
    @abstractmethod
    def _get_results(self, lst_output): #: list):
        """ 
        Get results of the different parts.

        Parameters:
            lst_output              : list     ; OUTPUT list with results
        """
        pass

    def get_device_max_ipc(self) -> float:
        """
        Get Max IPC of device

        Raises:
        """

        shell : Shell = Shell()
        compute_capability : str = shell.launch_command_show_all("nvcc ../src/measure_parts/compute_capability.cu --run", None)
        shell.launch_command("rm -f a.out", None) # delete 'a.out' generated
        if not compute_capability:
            raise ComputeCapabilityError
        dict_warps_schedulers_per_cc : dict = dict({3.0: 4, 3.2: 4, 3.5: 4, 3.7: 4, 5.0: 4, 5.2: 4, 5.3: 4, 
            6.0: 2, 6.1: 4, 6.2: 4, 7.0: 4, 7.5: 4, 8.0: 1}) 
        dict_ins_per_cycle : dict = dict({3.0: 1.5, 3.2: 1.5, 3.5: 1.5, 3.7: 1.5, 5.0: 1.5, 5.2: 1.5, 5.3: 1.5, 
            6.0: 1.5, 6.1: 1.5, 6.2: 1.5, 7.0: 1, 7.5: 1, 8.0: 1})
        return dict_warps_schedulers_per_cc.get(float(compute_capability))*dict_ins_per_cycle.get(float(compute_capability))
        pass
    
    def _add_result_part_to_lst(self, dict_values : dict, dict_desc : dict, 
        lst_to_add , isMetric : bool):
        """
        Add results of execution part (FrontEnd, BackEnd...) to list indicated by argument.

        Params:
            dict_values     : dict      ; diccionary with name_metric/event-value elements of the part to 
                                          add to 'lst_to_add'
            dict_desc       : dict      ; diccionary with name_metric/event-description elements of the 
                                          part to add to 'lst_to_add'
            lst_output      : list ; list where to add all elements
            isMetric        : bool      ; True if they are metrics or False if they are events

        Raises:
            MetricNoDefined             ; raised in case you have added an metric that is 
                                          not supported or does not exist in the NVIDIA analysis tool
            EventNoDefined              ; raised in case you have added an event that is 
                                          not supported or does not exist in the NVIDIA analysis tool
        """
        
        # metrics_events_not_average : list = LevelExecutionParameters.C_METRICS_AND_EVENTS_NOT_AVERAGE_COMPUTED.split(",")
        metrics_events_not_average  = LevelExecutionParameters.C_METRICS_AND_EVENTS_NOT_AVERAGE_COMPUTED.split(",")
        total_value : float = 0.0
        i : int = 0
        description : str 
        line_lenght : int 
        value_str : str
        total_value_str : str = ""
        if isMetric:
            metric_name : str
            description = "\t\t\t{:<45}{:<49}{:<5}".format('Metric Name','Metric Description', 'Value')
            line_lenght = len(description)
            value : float
            is_percentage : bool = False
            is_computed_as_average : bool 
            for key_value in dict_values:
                if dict_values[key_value][0][len(dict_values[key_value][0]) - 1] == "%":
                    is_percentage = True
                    # In NVIDIA scan tool, the percentages in each kernel are calculated on the total of 
                    # each kernel and not on the total of the application
                    if key_value in metrics_events_not_average:
                        raise ComputedAsAverageError(key_value)
                is_computed_as_average = not (key_value in metrics_events_not_average)
                total_value = round(self._get_total_value_of_list(dict_values[key_value], is_computed_as_average), 
                    LevelExecutionParameters.C_MAX_NUM_RESULTS_DECIMALS)
                if total_value.is_integer():
                    total_value = int(total_value)
                value_str = str(total_value)
                if is_percentage:
                    value_str += "%"
                    is_percentage = False
                metric_name = list(dict_desc.keys())[i]
                value_str = "\t\t\t{:<45}{:<49}{:<6} ".format(metric_name, dict_desc.get(metric_name), value_str) 
                if len(value_str) > line_lenght:
                    line_lenght = len(value_str)
                if i != len(dict_values) -1:
                    value_str += "\n"
                total_value_str += value_str
                i += 1
        else:
            event_name : str
            description = "\t\t\t{:<45}{:<46} {:<3}".format('Event Name','Event Description', 'Value')
            line_lenght = len(description)
            for key_value in dict_values:
                total_value = round(self._get_total_value_of_list(dict_values[key_value], False), 
                    LevelExecutionParameters.C_MAX_NUM_RESULTS_DECIMALS) # TODO eventos con decimales?
                if total_value.is_integer():
                    total_value = int(total_value)
                value_str = str(total_value)
                event_name = list(dict_desc.keys())[i]
                value_str = "\t\t\t{:<45}{:<47}{:<6} ".format(event_name, "-", value_str)
                if len(value_str) > line_lenght:
                    line_lenght = len(value_str)
                if i != len(dict_values) -1:
                    value_str += "\n"
                total_value_str += value_str
                i += 1
        spaces_lenght : int = len("\t\t\t")
        line_str : str = "\t\t\t" + f'{"-" * ((line_lenght - spaces_lenght) - 1)}'
        lst_to_add.append("\n" + line_str)
        lst_to_add.append(description)
        lst_to_add.append(line_str)
        lst_to_add.append(total_value_str)
        lst_to_add.append(line_str + "\n")
        pass

    def _get_total_value_of_list(self, list_values, computed_as_average : bool) -> float:
        """
        Get total value of list of metric/event
    
        Params:
            list_values         : list ; list to be computed
            computed_as_average : bool      ; True if you want to obtain total value as average
                                              as the average of the elements as a function of the 
                                              time executed or False if it is the total value per 
                                              increment
        Returns:
            Float with total value of the list
        """
        
        # TODO mirar este metodo, repetir for para no hacer if en cada iteraccion o que
        i : int = 0
        total_value : float = 0.0
        value_str : str = ""
        value : float = 0.0
        for value_str in list_values:
            if value_str[len(value_str) - 1] == "%":
                value = float(value_str[0:len(value_str) - 1])
                if computed_as_average:
                    value *= (self._percentage_time_kernel(i)/100.0)
                total_value += value
            else:
                if self.__compute_capability > TopDownParameters.C_COMPUTE_CAPABILITY_NVPROF_MAX_VALUE:
                    value = locale.atof(value_str)
                else:
                    value = float(value_str)
                if computed_as_average:
                    value *= (self._percentage_time_kernel(i)/100.0)
                total_value += value
            i += 1
        return total_value
        pass

    def _get_stalls_of_part(self, dict : dict) -> float:
        """
        Get percent of stalls of the dictionary indicated by argument.

        Params:
            dic :   dict    ; dictionary with stalls of the corresponding part

        Returns:
            A float with percent of stalls of the dictionary indicated by argument.
        """

        total_value : float = 0.0
        for key in dict.keys():
            total_value += self._get_total_value_of_list(dict.get(key), True)
        return total_value
        pass

    def recolect_metrics(self) -> bool:
        """
        Check if execution must recolect NVIDIA's scan tool metrics.

        Returns:
            Boolean with True if it has to recolect metrics or False if not
        """

        return self._recolect_metrics
        pass
    
    @abstractmethod
    def _percentage_time_kernel(self, kernel_number : int) -> float:
        """ 
        Get time percentage in each Kernel based on cycles elapsed metric/event name
        Each kernel measured is an index of dictionaries used by this program.

        Params:
                kernel_number                   : int   ; number of kernel
                cycles_elapsed_counter_name     : str   ; name of event/metric to obatin cycles elapsed in kernel
        Raises:
                ElapsedCyclesError      ; cycles elapsed in 'kernel_number' cannot be obtained
        """
        
        pass
    
