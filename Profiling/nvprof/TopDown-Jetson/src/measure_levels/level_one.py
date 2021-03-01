"""
Class that represents the level one of the execution

@author:    Alvaro Saiz (UC)
@date:      Jan-2021
@version:   1.0
"""

import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(1, parentdir) 
from measure_levels.level_execution import LevelExecution
from parameters.level_execution_params import LevelExecutionParameters
from shell.shell import Shell # launch shell arguments
from errors.level_execution_errors import *

class LevelOne(LevelExecution):
    """ 
    Class thath represents the level one of the execution.
    """

    def _launch(self) -> str:
        """ 
        Launch NVIDIA scan tool.
        
        Returns:
            String with results.

        Raises:
            ProfilingError      ; raised in case of error reading results from NVIDIA scan tool
        """

        shell : Shell = Shell()
        #output_file : str = self._output_file
        output_command : bool
        # if que muestra el resultado del NVPROF en el fichero
        #if output_file is None:
        #    output_command = shell.launch_command(self._generate_command(), LevelExecutionParameters.C_INFO_MESSAGE_EXECUTION_NVPROF)
        #else:
        #    output_command = shell.launch_command_redirect(self._generate_command(), LevelExecutionParameters.C_INFO_MESSAGE_EXECUTION_NVPROF, 
        #        output_file, True)
        output_command = shell.launch_command(self._generate_command(), LevelExecutionParameters.C_INFO_MESSAGE_EXECUTION_NVPROF)
        if output_command is None:
            raise ProfilingError
        return output_command  
        pass
    
    def _generate_command(self) -> str:
        """ 
        Generate command of execution with NVIDIA scan tool.

        Returns:
            String with command to be executed
        """

        command : str = ("sudo $(which nvprof) --metrics " + self._front_end.metrics_str() + 
            "," + self._back_end.metrics_str() + "," + self._divergence.metrics_str() + "," + self._extra_measure.metrics_str()
            + "," + self._retire.metrics_str() + "  --events " + self._front_end.events_str() + 
            "," + self._back_end.events_str() + "," + self._divergence.events_str() +  "," + self._extra_measure.events_str() +
             "," + self._retire.events_str() + " --unified-memory-profiling off " + self._program)
        return command
        pass

    def _get_results(self, lst_output : list[str]):
        """
        Get results of the different parts.

        Parameters:
            lst_output              : list[str]     ; OUTPUT list with results
        """

        #  Keep Results
        lst_output.append("\n\nList of counters/metrics measured according to the part.")
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
    