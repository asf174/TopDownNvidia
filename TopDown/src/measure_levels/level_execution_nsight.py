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
from measure_parts.extra_measure import ExtraMeasureNsight

class LevelExecutionNsight(LevelExecution, ABC):
    """ 
    Class that represents the levels of the execution with nsight scan tool
     
    Attributes:
        _extra_measure      : ExtraMeasureNsight  ; support measures
        _recolect_events    : bool          ; True if the execution must recolted the events used by NVIDIA scan tool
                                              or False in other case
    """

    def __init__(self, program : str, output_file : str, recoltect_metrics : bool):
        print("AKI SI ENTRO")
        self._extra_measure : ExtraMeasureNsight = ExtraMeasureNsight()
        super().__init__(program, output_file, recoltect_metrics)
        pass

    def extra_measure(self) -> ExtraMeasureNsight:
        """
        Return ExtraMeasureNsight part of the execution.

        Returns:
            reference to ExtraMeasureNsight part of the execution
        """
        
        return self._extra_measure
        pass

    @abstractmethod
    def run(self, lst_output : list):
        """
        Makes execution.
        
        Parameters:
            lst_output  : list ; list with results
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
    def _get_results(self, lst_output : list):
        """ 
        Get results of the different parts.

        Parameters:
            lst_output              : list     ; OUTPUT list with results
        """
        pass

    def extra_measure(self) -> ExtraMeasureNsight:
        """
        Return ExtraMeasureNsight part of the execution.

        Returns:
            reference to ExtraMeasureNsight part of the execution
        """

        return self._extra_measure
        pass

