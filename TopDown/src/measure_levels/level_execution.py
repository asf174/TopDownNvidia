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
from graph.pie_chart import PieChart

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
   
    @abstractmethod
    def __create_graph() -> PieChart:
        """ 
        Create a graph where figures are going to be saved.

        Returns:
            Referente to PieChart with graph
        """

        pass
   
    @abstractmethod
    def __add_graph_data(self, graph : PieChart):
        """ 
        Add data to graph.

        Params:
            graph   : PieChart  ; reference to PieChart where save figures        
        """
        
        pass
       
    @abstractmethod    
    def showGraph(self):
        """Show graph to see results."""
        
        pass
    
    @abstractmethod    
    def saveGraph(self, file : str):
        """ 
        Save graph in file indicated as argument.
        
        Params:
            file    : str   ; path to output file where save fig
        """
        
        pass

