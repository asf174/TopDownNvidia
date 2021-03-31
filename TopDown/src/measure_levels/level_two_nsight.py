import re
from measure_levels.level_two import LevelTwo
from measure_levels.level_one_nsight import LevelOneNsight
from measure_parts.back_core_bound import BackCoreBoundNsight
from measure_parts.back_memory_bound import BackMemoryBoundNsight
from measure_parts.front_band_width import FrontBandWidthNsight
from measure_parts.front_dependency import FrontDependencyNsight
from show_messages.message_format import MessageFormat
from errors.level_execution_errors import *


class LevelTwoNsight(LevelTwo, LevelOneNsight):
    """
    Class with level two of the execution based on NSIGHT scan tool
    
    Atributes:
        _back_memory_bound      : BackMemoryBoundNsight     ; backs'  memory bound part
        _back_core_bound        : BackCoreBoundNsight       ; backs' core bound part
        _front_band_width       : FrontBandWidthNsight      ; front's bandwith part
        _front_dependency       : FrontDependencyNsight     ; front's dependency part
    """

    def __init__(self, program : str, output_file : str, recoltect_metrics : bool):
        
        self._back_core_bound : BackCoreBoundNsight = BackCoreBoundNsight()
        self._back_memory_bound : BackMemoryBoundNsight = BackMemoryBoundNsight()
        self._front_band_width : FrontBandWidthNsight = FrontBandWidthNsight()
        self._front_dependency : FrontDependencyNsight = FrontDependencyNsight()
        super().__init__(program, output_file, recoltect_metrics)
        pass

    def back_core_bound(self) -> BackCoreBoundNsight:
        """
        Return CoreBound part of the execution.

        Returns:
            reference to CoreBound part of the execution
        """
        
        return self._back_core_bound
        pass

    def back_memory_bound(self) -> BackMemoryBoundNsight:
        """
        Return MemoryBoundNsight part of the execution.

        Returns:
            reference to MemoryBoundNsight part of the execution
        """
        
        return self._back_memory_bound
        pass

    def front_band_width(self) -> FrontBandWidthNsight:
        """
        Return FrontBandWidthNsight part of the execution.

        Returns:
            reference to FrontBandWidthNsight part of the execution
        """
        
        return self._front_band_width
        pass

    def front_dependency(self) -> FrontDependencyNsight:
        """
        Return FrontDependencyNsight part of the execution.

        Returns:
            reference to FrontDependencyNsight part of the execution
        """
        
        return self._front_dependency
        pass

    def _generate_command(self) -> str:
        """ 
        Generate command of execution with NVIDIA scan tool.

        Returns:
            String with command to be executed
        """
        
        command : str = ("sudo $(which ncu) --metrics " + self._front_end.metrics_str() +
            "," + self._back_end.metrics_str() + "," + self._divergence.metrics_str() + "," + self._extra_measure.metrics_str()
            + "," + self._retire.metrics_str() + "," + self._back_core_bound.metrics_str() + "," + self._back_memory_bound.metrics_str() +
            " " + self._program)

        return command
        pass

    def _get_results(self, lst_output : list):
        """ 
        Get results of the different parts.

        Parameters:
            lst_output              : list     ; OUTPUT list with results
        """

        # revisar en unos usa atributo y en otros la llamada al metodo
        #  Keep Results
        converter : MessageFormat = MessageFormat()
        if not self._recolect_metrics:
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
        is_defined_front_end_value : str = super().front_end().get_metric_value(metric_name)
        is_defined_front_end_unit : str = super().front_end().get_metric_unit(metric_name)

        is_defined_back_end_value : str = super().back_end().get_metric_value(metric_name)
        is_defined_back_end_unit : str = super().back_end().get_metric_unit(metric_name)

        is_defined_divergence_value : str = super().divergence().get_metric_value(metric_name)
        is_defined_divergence_unit : str = super().divergence().get_metric_unit(metric_name)

        is_defined_retire_value : str = super().retire().get_metric_value(metric_name)
        is_defined_retire_unit : str = super().retire().get_metric_unit(metric_name)

        is_defined_extra_measure_value : str = super().extra_measure().get_metric_value(metric_name)
        is_defined_extra_measure_unit : str = super().extra_measure().get_metric_unit(metric_name)
        
        # comprobar el "if not"
        if not ((is_defined_front_end_value is None and is_defined_front_end_unit is None)
            or (is_defined_back_end_value is None and is_defined_back_end_unit is None)
            or (is_defined_divergence_value is None and is_defined_divergence_unit is None)
            or (is_defined_retire_value is None or is_defined_retire_unit is None)
            or (is_defined_extra_measure_value is None and is_defined_extra_measure_unit is None)):
            return False
        return True
        pass

    def _set_memory_core_bandwith_dependency_results(self, results_launch : str):
        """
        Set results of the level two part (that are not level one).
        
        Params:
            results_launch : str   ; results generated by NVIDIA scan tool.
            
        Raises:
            MetricNotAsignedToPart ; raised if some metric is found don't assigned 
                                     to any measure part
        """

        metric_name : str
        metric_unit : str
        metric_value : str
        line : str
        i : int
        list_words : list
        back_core_bound_value_has_found : bool
        back_core_bound_unit_has_found : bool
        back_memory_bound_value_has_found : bool
        back_memory_bound_unit_has_found : bool
        front_band_width_value_has_found : bool
        front_band_width_unit_has_found : bool
        front_dependency_value_has_found : bool
        front_dependency_unit_has_found : bool
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
                
                back_core_bound_value_has_found = self._back_core_bound.set_metric_value(metric_name, metric_value)
                back_core_bound_unit_has_found = self._back_core_bound.set_metric_unit(metric_name, metric_unit)
                back_memory_bound_value_has_found = self._back_memory_bound.set_metric_value(metric_name, metric_value)
                back_memory_bound_unit_has_found = self._back_memory_bound.set_metric_unit(metric_name, metric_unit)
                front_band_width_value_has_found = self._front_band_width.set_metric_value(metric_name, metric_value)
                front_band_width_unit_has_found = self._front_band_width.set_metric_unit(metric_name, metric_unit)
                front_dependency_value_has_found = self._front_dependency.set_metric_value(metric_name, metric_value)
                front_dependency_unit_has_found = self._front_dependency.set_metric_unit(metric_name, metric_unit)
                if (not (back_core_bound_value_has_found or back_memory_bound_value_has_found or front_band_width_value_has_found
                    or front_dependency_value_has_found) or not(back_core_bound_unit_has_found or back_memory_bound_unit_has_found
                    or front_band_width_unit_has_found or front_dependency_unit_has_found)):
                    if not self._metricExists(metric_name):
                        raise MetricNotAsignedToPart(metric_name)
        pass



