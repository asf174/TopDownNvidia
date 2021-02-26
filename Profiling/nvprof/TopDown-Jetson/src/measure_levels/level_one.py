"""
Class that represents the level one of the execution

@author:    Alvaro Saiz (UC)
@date:      Jan-2021
@version:   1.0
"""

from abc import abstractmethod
import sys
path : str = "/home/alvaro/Documents/Facultad/"
path_desp : str = "/mnt/HDD/alvaro/"
sys.path.insert(1, path + "TopDownNvidia/Profiling/nvprof/TopDown-Jetson/src/errors")
sys.path.insert(1,  path + "TopDownNvidia/Profiling/nvprof/TopDown-Jetson/src/parameters")
sys.path.insert(1,  path + "TopDownNvidia/Profiling/nvprof/TopDown-Jetson/src/measure_parts")
sys.path.insert(1,  path + "TopDownNvidia/Profiling/nvprof/TopDown-Jetson/src/measure_levels")
from level_execution import LevelExecution
from level_execution_params import LevelExecutionParameters
from shell.shell import Shell # launch shell arguments
from level_execution_errors import *

class LevelOne(LevelExecution):
    """ 
    Class thath represents the level one of the execution.
    """
    
    def __init__(self, program : str, output_file : str):
        super().__init__(program, output_file)

    def _launch(self) -> str:
        """ 
        Launch NVIDIA scan tool.
        
        Returns:
            String with results.

        Raises:
            ProfilingError      ; raised in case of error reading results from NVIDIA scan tool
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
        return output_command  
        pass

    def _get_results(self, lst_output : list[str]):
        """ 
        Get results of the different parts.

        Parameters:
            lst_output              : list[str]     ; OUTPUT list with results
        """

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
        
        output_command : str = self._launch()
        super()._set_front_back_divergence_retire_results(output_command)
        self._get_results(lst_output)
        pass
    