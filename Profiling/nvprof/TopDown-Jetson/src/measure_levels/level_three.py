"""
Class that represents the level three of the execution

@author:    Alvaro Saiz (UC)
@date:      Jan-2021
@version:   1.0
"""

import re
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from measure_levels.level_execution import LevelTwo
from parameters.level_execution_params import LevelExecutionParameters
from shell.shell import Shell # launch shell arguments
from errors.level_execution_errors import *

class LevelThree(LevelTwo):
    """
    Class with level three of the execution.
    
    Atributes:
        __constant_memory_bound     : ConstantMemoryBound   ; constant cache part
    """

    def __init__(self, program : str, output_file : str):
        
        self.__constant_memory_bound : ConstantMemoryBound = ConstantMemoryBound()
        super().__init__(program, output_file)
        pass

    def constant_memory_bound() -> str:
        """
        Return ConstantMemoryBound part of the execution.

        Returns:
            reference to ConstantMemoryBound part of the execution
        """

        return self.__constant_memory_bound


    def _generate_command(self) -> str:
        """ 
        Generate command of execution with NVIDIA scan tool.

        Returns:
            String with command to be executed
        """

        pass

    def run(self, lst_output : list[str]):
        """
        Makes execution.
        
        Parameters:
            lst_output  : list[str] ; list with results
        """
        
        pass
