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

    def __init__(self, program : str, output_file : str, recoltect_metrics : bool):
        
        self.__memory_constant_memory_bound : MemoryConstantMemoryBoundNsight = MemoryConstantMemoryBoundNsight()
        super().__init__(program, output_file, recoltect_metrics)
        pass

    def memory_constant_memory_bound(self) -> MemoryConstantMemoryBoundNsight:
        """
        Return MemoryConstantMemoryBoundNsight part of the execution.

        Returns:
            reference to MemoryConstantMemoryBoundNsight part of the execution
        """

        return self.__memory_constant_memory_bound

    def _generate_command(self) -> str:
        """ 
        Generate command of execution with NVIDIA scan tool.

        Returns:
            String with command to be executed
        """

        command : str = ("sudo $(which ncu) --metrics " + self._front_end.metrics_str() +
            "," + self._back_end.metrics_str() + "," + self._divergence.metrics_str() + "," +
            self._extra_measure.metrics_str() + "," + self._retire.metrics_str() + "," +
            self._back_core_bound.metrics_str() + "," + self._back_memory_bound.metrics_str() +
            " " + self._program)
        return command
        pass

    def _get_results(self, lst_output : list):
        """ 
        Get results of the different parts. TODO

        Parameters:
            lst_output              : list     ; OUTPUT list with results
        """

        # revisar en unos usa atributo y en otros la llamada al metodo TODO
        #  Keep Results
        converter : MessageFormat = MessageFormat()

        if not self._recolect_metrics:
            return
        if self._recolect_metrics and self._front_end.metrics_str() != "":
            lst_output.append(converter.underlined_str(self._front_end.name()))
            super()._add_result_part_to_lst(self._front_end.metrics(), 
                self._front_end.metrics_description(), lst_output)
        if  self._recolect_metrics and self._front_band_width.metrics_str() != "":
            lst_output.append(converter.underlined_str(self._front_band_width.name()))
            super()._add_result_part_to_lst(self._front_band_width.metrics(), 
                self._front_band_width.metrics_description(), lst_output)      
        if self._recolect_metrics and self._front_dependency.metrics_str() != "":
            lst_output.append(converter.underlined_str(self._front_dependency.name()))
            super()._add_result_part_to_lst(self._front_dependency.metrics(), 
                self._front_dependency.metrics_description(), lst_output)
        if self._recolect_metrics and self._back_end.metrics_str() != "":
            lst_output.append(converter.underlined_str(self._back_end.name()))
            super()._add_result_part_to_lst(self._back_end.metrics(), 
                self._back_end.metrics_description(), lst_output)
        if self._recolect_metrics and self._back_core_bound.metrics_str() != "":
            lst_output.append(converter.underlined_str(self._back_core_bound.name()))
            super()._add_result_part_to_lst(self._back_core_bound.metrics(), 
                self._back_core_bound.metrics_description(), lst_output)
        if self._recolect_metrics and self._back_memory_bound.metrics_str() != "":
            lst_output.append(converter.underlined_str(self._back_memory_bound.name()))
            super()._add_result_part_to_lst(self._back_memory_bound.metrics(), 
                self._back_memory_bound.metrics_description(), lst_output)
        if  self._recolect_metrics and self.__memory_constant_memory_bound.metrics_str() != "":
            lst_output.append(converter.underlined_str(self.__memory_constant_memory_bound.name()))
            super()._add_result_part_to_lst(self.__memory_constant_memory_bound.metrics(), 
                self.__memory_constant_memory_bound.metrics_description(), lst_output)
        if self._recolect_metrics and self._divergence.metrics_str() != "":
            lst_output.append(converter.underlined_str(self._divergence.name()))
            super()._add_result_part_to_lst(self._divergence.metrics(), 
                self._divergence.metrics_description(), lst_output)
        if self._recolect_metrics and  self._retire.metrics_str() != "":
                lst_output.append(converter.underlined_str(self._retire.name()))
                super()._add_result_part_to_lst(self._retire.metrics(), 
                self._retire.metrics_description(), lst_output)
        if self._recolect_metrics and self._extra_measure.metrics_str() != "":
            lst_output.append(converter.underlined_str(self._extra_measure.name()))
            super()._add_result_part_to_lst(self._extra_measure.metrics(), 
                self._extra_measure.metrics_description(), lst_output)
        lst_output.append("\n")
        pass
    pass

    def _set_constant_memory_bound_results(self, results_launch : str):
        """
        Set results of the level three part (that are not level one or two).
        
        Params:
            results_launch : str   ; results generated by NVIDIA scan tool.
        """
        
        metric_name : str
        metric_unit : str
        metric_value : str
        line : str
        i : int
        list_words : list
        memory_constant_memory_bound_value_has_found : bool
        memory_constant_memory_bound_unit_has_found : bool
        can_read_results : bool = False
        for line in str(results_launch).splitlines():
            line = re.sub(' +', ' ', line) # delete more than one spaces and put only one
            list_words = line.split(" ")
            # Check if it's line of interest:
            # ['', 'metric_name','metric_unit', 'metric_value']
            if not can_read_results:
                if list_words[0] == "==PROF==" and list_words[1] == "Disconnected":
                        can_read_results = True
                continue
            if (len(list_words) == 4 or len(list_words) == 3) and list_words[1][0] != "-":
                if len(list_words) == 3:
                    metric_name = list_words[1]
                    metric_unit = ""
                    metric_value = list_words[2]
                else:
                    metric_name = list_words[1]
                    metric_unit = list_words[2]
                    metric_value = list_words[3]

                front_dependency_value_has_found = self._front_dependency.set_metric_value(metric_name, metric_value)
                front_dependency_unit_has_found = self._front_dependency.set_metric_unit(metric_name, metric_unit)
                if not memory_constant_memory_bound_value_has_found or not memory_constant_memory_bound_unit_has_found:
                    if not self._metricExists(metric_name):
                        raise MetricNotAsignedToPart(metric_name)
        pass

    def _metricExists(self, metric_name : str) -> bool:
        """
        Check if metric exists in some part of the execution (MemoryBound, CoreBound...). 

        Params:
            metric_name  : str   ; name of the metric to be checked

        Returns:
            True if metric is defined in some part of the execution (MemoryBound, CoreBound...)
            or false in other case
        """

        # revisar si las superiores (core bound sobre back end) tiene que estar definida
        is_defined_front_dependency_value : str = super().front_dependency().get_metric_value(metric_name)
        is_defined_front_dependency_unit : str = super().front_dependency().get_metric_unit(metric_name)

        is_defined_front_band_width_value : str = super().front_band_width().get_metric_value(metric_name)
        is_defined_front_band_width_unit : str = super().front_band_width().get_metric_unit(metric_name)

        is_defined_back_memory_bound_value : str = super().back_memory_bound().get_metric_value(metric_name)
        is_defined_back_memory_bound_unit : str = super().back_memory_bound().get_metric_unit(metric_name)

        is_defined_back_core_bound_value : str = super().back_core_bound().get_metric_value(metric_name)
        is_defined_back_core_bound_unit : str = super().back_core_bound().get_metric_unit(metric_name)


        # comprobar el "if not"
        if not ((is_defined_front_dependency_value is None and is_defined_front_dependency_unit is None)
            or (is_defined_front_band_width_value is None and is_defined_front_band_width_unit is None)
            or (is_defined_back_memory_bound_value is None and is_defined_back_memory_bound_unit is None)
            or (is_defined_back_core_bound_value is None or is_defined_back_core_bound_unit is None)):
            return False
        return True

    def _set_memory_constant_memory_bound_results(self, results_launch : str):
        """
        Set results of the level three part (that are not level one or two).
        
        Params:
            results_launch : str   ; results generated by NVIDIA scan tool.
        """
        
        metric_name : str
        metric_unit : str
        metric_value : str
        line : str
        i : int
        list_words : list
        memory_constant_memory_bound_value_has_found: bool
        memory_constant_memory_bound_unit_has_found : bool
        can_read_results : bool = False
        for line in str(results_launch).splitlines():
            line = re.sub(' +', ' ', line) # delete more than one spaces and put only one
            list_words = line.split(" ")
            # Check if it's line of interest:
            # ['', 'metric_name','metric_unit', 'metric_value']
            if not can_read_results:
                if list_words[0] == "==PROF==" and list_words[1] == "Disconnected":
                        can_read_results = True
                continue
            if (len(list_words) == 4 or len(list_words) == 3) and list_words[1][0] != "-":
                if len(list_words) == 3:
                    metric_name = list_words[1]
                    metric_unit = ""
                    metric_value = list_words[2]
                else:
                    metric_name = list_words[1]
                    metric_unit = list_words[2]
                    metric_value = list_words[3]

                memory_constant_memory_bound_value_has_found = self.__memory_constant_memory_bound.set_metric_value(metric_name, metric_value)
                memory_constant_memory_bound_unit_has_found = self.__memory_constant_memory_bound.set_metric_unit(metric_name, metric_unit)
                if not memory_constant_memory_bound_value_has_found or not memory_constant_memory_bound_unit_has_found:
                    if not self._metricExists(metric_name):
                        raise MetricNotAsignedToPart(metric_name)
        pass

