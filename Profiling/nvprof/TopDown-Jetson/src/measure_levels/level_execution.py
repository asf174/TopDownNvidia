"""
Class that represents the levels of the execution.

@author:    Alvaro Saiz (UC)
@date:      Jan-2021
@version:   1.0
"""

from abc import ABC, abstractmethod # abstract class
import re
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from measure_parts.extra_measure import ExtraMeasure    
from shell.shell import Shell # launch shell arguments
from parameters.level_execution_params import LevelExecutionParameters # parameters of program
from errors.level_execution_errors import *

class LevelExecution(ABC):
    """ 
    Class that represents the levels of the execution.
     
    Attributes:
        _extra_measure  : ExtraMeasure  ; support measures
        _output_file    : str           ; path to output file with results. 'None' to don't use
                                          output file
        _program        : str           ; program of the execution
    """

    def __init__(self, program : str, output_file : str):
        self._extra_measure : ExtraMeasure = ExtraMeasure()
        self._program = program
        self._output_file : str = output_file
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
    def run(self, lst_output : list[str]):
        """
        Makes execution.
        
        Parameters:
            lst_output  : list[str] ; list with results
        """
        
        pass

    @abstractmethod
    def _get_results(self, lst_output : list[str]):
        """ 
        Get results of the different parts.

        Parameters:
            lst_output              : list[str]     ; OUTPUT list with results
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
    def _get_results(self, lst_output : list[str]):
        """ 
        Get results of the different parts.

        Parameters:
            lst_output              : list[str]     ; OUTPUT list with results
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
    
    def _add_result_part_to_lst(self, dict_values : dict, dict_desc : dict, message : str, 
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
        #lst_to_add.append("\n")
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
            value_event : str 
            for event_name in dict_values:
                value_event = dict_values.get(event_name)
                if event_name is None or value_event is None:
                    #print(str(event_name) + " " + str(value_event))
                    raise EventNoDefined(event_name)
                lst_to_add.append("\t\t\t{:<45} {:<47} {:<6} ".format(event_name, "-", value_event))
        lst_to_add.append("\t\t\t----------------------------------------------------"
            +"---------------------------------------------------")
        pass

    def extra_measure(self) -> ExtraMeasure:
        """
        Return ExtraMeasure part of the execution.

        Returns:
            reference to ExtraMeasure part of the execution
        """
        
        return self._extra_measure
        pass

    def __get__key_max_value(self, dictionary : dict) -> str:
        """ 
        Find key with highest value in dictionary.

        Params:
            dictionary : dict ; dictionary to check

        Returns:
            String with the key with the highest value
            or 'None' if dictionary is empty
        """

        max_key : str = None
        max_value : float = float('-inf')
        value : float
        for key in dictionary.keys():
            value = float(dictionary.get(key)[0 : len(dictionary.get(key)) - 1])
            if value > max_value:
                max_key = key
                max_value = value
        return max_key
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
        value_str : str 
        for key in dict.keys():
                value_str = dict.get(key)
                if value_str[len(value_str) - 1] == "%":  # check percenttage
                    total_value += float(value_str[0 : len(value_str) - 1])
        return total_value
        pass
