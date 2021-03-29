from abc import ABC, abstractmethod # abstract class
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from measure_parts.extra_measure import ExtraMeasure    
from shell.shell import Shell # launch shell arguments
from parameters.level_execution_params import LevelExecutionParameters # parameters of program
from errors.level_execution_errors import *
from measure_levels.level_execution import LevelExecution 
from measure_parts.extra_measure import ExtraMeasureNvprof

class LevelExecutionNvprof(LevelExecution, ABC):
    """ 
    Class that represents the levels of the execution with nvprof scan tool
     
    Attributes:
         _extra_measure      : ExtraMeasureNvprof  ; support measures
        _recolect_events    : bool          ; True if the execution must recolted the events used by NVIDIA scan tool
                                              or False in other case
    """

    def __init__(self, program : str, output_file : str, recoltect_metrics : bool, recolect_events : bool):
        self._extra_measure : ExtraMeasureNvprof = ExtraMeasureNvprof()
        self._recolect_events = recolect_events
        super().__init__(program, output_file, recoltect_metrics)
        pass

    def recolect_events(self) -> bool:
        """
        Check if execution must recolect NVIDIA's scan tool events.

        Returns:
            Boolean with True if it has to recolect events or False if not
        """

        return self._recolect_events
        pass

    def extra_measure(self) -> ExtraMeasureNvprof:
        """
        Return ExtraMeasureNvprof part of the execution.

        Returns:
            reference to ExtraMeasureNvprof part of the execution
        """
        
        return self._extra_measure
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
    def _generate_command(self) -> str:
        """ 
        Generate command of execution with NVIDIA scan tool.

        Returns:
            String with command to be executed
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

    def _get_cycles_elaspsed_per_kernel(self, kernel_number : int):
        """ 
        Get cycles elapsed per kernel.

        Params:
            kernel_number   : int   ; number of kernel
        """
        ## mirar porque lo estoy comprobando cada vez que quiero el indice TODO

        if not LevelExecutionParameters.C_CYCLES_ELAPSED_NAME in self._extra_measure.events():
            raise ElapsedCyclesError
        return self._extra_measure.get_event_value(LevelExecutionParameters.C_CYCLES_ELAPSED_NAME)[kernel_number]
        pass

    def _get_percentage_time(self, kernel_number : int) -> float:
        """ 
        Get time percentage in each Kernel.
        Each kernel measured is an index of dictionaries used by this program.

        Params:
            kernel_number   : int   ; number of kernel
        """
        #TODO lanzar excepcion
        value_lst : list[str] = self._extra_measure.get_event_value(LevelExecutionParameters.C_CYCLES_ELAPSED_EVENT_NAME_NVPROF)
        if value_lst is None:
            raise ElapsedCyclesError
        value_str : str
        total_value : float = 0.0
        for value_str in value_lst:
            total_value += float(value_str)
        return (float(value_lst[kernel_number])/total_value)*100.0
        pass
