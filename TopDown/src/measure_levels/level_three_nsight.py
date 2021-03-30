"""
Class that represents the level three of the execution based 
on NSIGHT scan tool.

@author:    Alvaro Saiz (UC)
@date:      Jan-2021
@version:   1.0
"""

import re
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(1, parentdir) 
from measure_levels.level_two_nsight import LevelTwoNsight
from measure_parts.memory_constant_memory_bound import MemoryConstantMemoryBoundNsight
from measure_levels.level_three import LevelThree
from show_messages.message_format import MessageFormat

class LevelThreeNsight(LevelThree, LevelTwoNsight):
    """
    Class with level three of the execution based on Nsight scan tool
    
    Atributes:
        __memory_constant_memory_bound     : ConstantMemoryBoundNsight   ; constant cache part
    """

    def __init__(self, program : str, output_file : str, recoltect_metrics : bool, recolect_events : bool, is_tesla_device : bool):
        
        self.__constant_memory_bound : ConstantMemoryBoundNsight = ConstantMemoryBoundNsight()
        super().__init__(program, output_file, recoltect_metrics, recolect_events)
        pass

    def constant_memory_bound(self) -> MemoryConstantMemoryBoundNsight:
        """
        Return ConstantMemoryBoundNsight part of the execution.

        Returns:
            reference to ConstantMemoryBoundNsight part of the execution
        """

        return self.__memory_constant_memory_bound
    
    def _get_results(self, lst_output : list):
        """ 
        Get results of the different parts. TODO

        Parameters:
            lst_output              : list     ; OUTPUT list with results
        """

        # revisar en unos usa atributo y en otros la llamada al metodo TODO
        #  Keep Results
        converter : MessageFormat = MessageFormat()

        if not self._recolect_metrics and not self._recolect_events:
            return
        if self._recolect_metrics and self._front_end.metrics_str() != "":
            lst_output.append(converter.underlined_str(self._front_end.name()))
            super()._add_result_part_to_lst(self._front_end.metrics(), 
                self._front_end.metrics_description(), lst_output, True)
        if  self._recolect_metrics and self._front_band_width.metrics_str() != "":
            lst_output.append(converter.underlined_str(self._front_band_width.name()))
            super()._add_result_part_to_lst(self._front_band_width.metrics(), 
                self._front_band_width.metrics_description(), lst_output, True)      
        if self._recolect_metrics and self._front_dependency.metrics_str() != "":
            lst_output.append(converter.underlined_str(self._front_dependency.name()))
            super()._add_result_part_to_lst(self._front_dependency.metrics(), 
                self._front_dependency.metrics_description(), lst_output, True)
        if self._recolect_metrics and self._back_end.metrics_str() != "":
            lst_output.append(converter.underlined_str(self._back_end.name()))
            super()._add_result_part_to_lst(self._back_end.metrics(), 
                self._back_end.metrics_description(), lst_output, True)
        if self._recolect_metrics and self._back_core_bound.metrics_str() != "":
            lst_output.append(converter.underlined_str(self._back_core_bound.name()))
            super()._add_result_part_to_lst(self._back_core_bound.metrics(), 
                self._back_core_bound.metrics_description(), lst_output, True)
        if self._recolect_metrics and self._back_memory_bound.metrics_str() != "":
            lst_output.append(converter.underlined_str(self._back_memory_bound.name()))
            super()._add_result_part_to_lst(self._back_memory_bound.metrics(), 
                self._back_memory_bound.metrics_description(), lst_output, True)
        if  self._recolect_metrics and self.__constant_memory_bound.metrics_str() != "":
            lst_output.append(converter.underlined_str(self.__constant_memory_bound.name()))
            super()._add_result_part_to_lst(self.__constant_memory_bound.metrics(), 
                self.__constant_memory_bound.metrics_description(), lst_output, True)
        if self._recolect_metrics and self._divergence.metrics_str() != "":
            lst_output.append(converter.underlined_str(self._divergence.name()))
            super()._add_result_part_to_lst(self._divergence.metrics(), 
                self._divergence.metrics_description(), lst_output, True)
        if self._recolect_metrics and  self._retire.metrics_str() != "":
                lst_output.append(converter.underlined_str(self._retire.name()))
                super()._add_result_part_to_lst(self._retire.metrics(), 
                self._retire.metrics_description(), lst_output, True)
        if self._recolect_metrics and self._extra_measure.metrics_str() != "":
            lst_output.append(converter.underlined_str(self._extra_measure.name()))
            super()._add_result_part_to_lst(self._extra_measure.metrics(), 
                self._extra_measure.metrics_description(), lst_output, True)
        lst_output.append("\n")
        pass
    pass