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
from measure_parts.back_core_bound import BackCoreBoundNsight
from measure_parts.back_memory_bound import BackMemoryBoundNsight
from measure_parts.front_band_width import FrontBandWidthNsight
from measure_parts.front_dependency import FrontDependencyNsight
from measure_parts.front_end import FrontEndNsight
from measure_parts.back_end import BackEndNsight
from measure_parts.divergence import DivergenceNsight
from measure_parts.retire import RetireNsight
from measure_parts.extra_measure import ExtraMeasureNsight
from measure_parts.memory_mio_throttle import MemoryMioThrottleNsight
from measure_parts.memory_tex_throttle import MemoryTexThrottleNsight
from show_messages.message_format import MessageFormat
from parameters.memory_constant_memory_bound_params import MemoryConstantMemoryBoundParameters
from parameters.memory_mio_throttle_params import MemoryMioThrottleParameters
from parameters.memory_tex_throttle_params import MemoryTexThrottleParameters

class LevelThreeNsight(LevelThree, LevelTwoNsight):
    """
    Class with level three of the execution based on Nsight scan tool
    
    Atributes:
        __memory_constant_memory_bound      : ConstantMemoryBoundNsight ; constant cache part
        __memory_mio_throttle               : MemoryMioThrottleNsight   ; mio throttle part
        __memory_tex_throttle               : MemoryTexThrottleNsight   ; tex throttle part    
    """

    def __init__(self, program : str, input_file : str, output_file : str, output_scan_file : str, collect_metrics : bool,
        front_end : FrontEndNsight, back_end : BackEndNsight, divergence : DivergenceNsight, retire : RetireNsight,
        extra_measure : ExtraMeasureNsight, front_band_width : FrontBandWidthNsight, front_dependency : FrontDependencyNsight,
        back_core_bound : BackCoreBoundNsight, back_memory_bound : BackMemoryBoundNsight):

        self.__memory_constant_memory_bound : MemoryConstantMemoryBoundNsight = MemoryConstantMemoryBoundNsight(
            MemoryConstantMemoryBoundParameters.C_MEMORY_CONSTANT_MEMORY_BOUND_NAME, 
            MemoryConstantMemoryBoundParameters.C_MEMORY_CONSTANT_MEMORY_BOUND_DESCRIPTION, 
            MemoryConstantMemoryBoundParameters.C_MEMORY_CONSTANT_MEMORY_BOUND_NSIGHT_METRICS)

        self. __memory_mio_throttle : MemoryMioThrottleNsight =  MemoryMioThrottleNsight (
            MemoryMioThrottleParameters.C_MEMORY_MIO_THROTTLE_NAME, 
            MemoryMioThrottleParameters.C_MEMORY_MIO_THROTTLE_DESCRIPTION, 
            MemoryMioThrottleParameters.C_MEMORY_MIO_THROTTLE_NSIGHT_METRICS)

        self. __memory_mio_throttle : MemoryTexThrottleNsight =  MemoryTexThrottleNsight (
            MemoryTexThrottleParameters.C_MEMORY_TEX_THROTTLE_NAME, 
            MemoryTexThrottleParameters.C_MEMORY_TEX_THROTTLE_DESCRIPTION, 
            MemoryTexThrottleParameters.C_MEMORY_TEX_THROTTLE_NSIGHT_METRICS)

        super().__init__(program, input_file, output_file, output_scan_file, collect_metrics, front_end, back_end, divergence, 
        retire, extra_measure, front_band_width, front_dependency, back_core_bound, back_memory_bound)
        pass  

    def memory_constant_memory_bound(self) -> MemoryConstantMemoryBoundNsight:
        """
        Return MemoryConstantMemoryBoundNsight part of the execution.

        Returns:
            reference to MemoryConstantMemoryBoundNsight part of the execution
        """

        return self.__memory_constant_memory_bound
        pass
    
    def memory_mio_throttle(self) -> MemoryMioThrottleNsight:
        """
        Return MemoryMioThrottleNsight part of the execution.

        Returns:
            reference to MemoryMioThrottleNsight part of the execution
        """

        return self.__memory_mio_throttle
        pass
  
    def memory_tex_throttle(self) -> MemoryTexThrottleNsight:
        """
        Return MemoryTexThrottleNsight part of the execution.

        Returns:
            reference to MemoryTexThrottleNsight part of the execution
        """

        return self.__memory_tex_throttle
        pass


    def _generate_command(self) -> str:
        """ 
        Generate command of execution with NVIDIA scan tool.

        Returns:
            String with command to be executed
        """

        command : str = ("ncu --target-processes all --metrics " + self._front_end.metrics_str() +
            "," + self._back_end.metrics_str() + "," + self._divergence.metrics_str() + "," +
            self._extra_measure.metrics_str() + "," + self._retire.metrics_str() + "," + 
            self._front_band_width.metrics_str() + "," + self._front_dependency.metrics_str() + 
            "," + self._back_core_bound.metrics_str() + "," + self._back_memory_bound.metrics_str() +
             " " + self._program)
        return command
        pass
    
    def set_results(self,output_command : str):
        """
        Set results of execution ALREADY DONE. Results are in the argument.

        Params:
            output_command : str    ; str with results of execution.
        """

        super()._set_front_back_divergence_retire_results(output_command) # level one results
        super()._set_memory_core_bandwith_dependency_results(output_command) # level two
        self._set_memory_constant_memory_bound_mio_tex_throttle_results(output_command) # level three
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

        if not self._collect_metrics:
            return
        if self._collect_metrics and self._front_end.metrics_str() != "":
            lst_output.append(converter.underlined_str(self._front_end.name()))
            super()._add_result_part_to_lst(self._front_end.metrics(), 
                self._front_end.metrics_description(), lst_output)
        if  self._collect_metrics and self._front_band_width.metrics_str() != "":
            lst_output.append(converter.underlined_str(self._front_band_width.name()))
            super()._add_result_part_to_lst(self._front_band_width.metrics(), 
                self._front_band_width.metrics_description(), lst_output)      
        if self._collect_metrics and self._front_dependency.metrics_str() != "":
            lst_output.append(converter.underlined_str(self._front_dependency.name()))
            super()._add_result_part_to_lst(self._front_dependency.metrics(), 
                self._front_dependency.metrics_description(), lst_output)
        if self._collect_metrics and self._back_end.metrics_str() != "":
            lst_output.append(converter.underlined_str(self._back_end.name()))
            super()._add_result_part_to_lst(self._back_end.metrics(), 
                self._back_end.metrics_description(), lst_output)
        if self._collect_metrics and self._back_core_bound.metrics_str() != "":
            lst_output.append(converter.underlined_str(self._back_core_bound.name()))
            super()._add_result_part_to_lst(self._back_core_bound.metrics(), 
                self._back_core_bound.metrics_description(), lst_output)
        if self._collect_metrics and self._back_memory_bound.metrics_str() != "":
            lst_output.append(converter.underlined_str(self._back_memory_bound.name()))
            super()._add_result_part_to_lst(self._back_memory_bound.metrics(), 
                self._back_memory_bound.metrics_description(), lst_output)
        if  self._collect_metrics and self.__memory_constant_memory_bound.metrics_str() != "":
            lst_output.append(converter.underlined_str(self.__memory_constant_memory_bound.name()))
            super()._add_result_part_to_lst(self.__memory_constant_memory_bound.metrics(), 
                self.__memory_constant_memory_bound.metrics_description(), lst_output)
        if  self._collect_metrics and self.__memory_mio_throttle.metrics_str() != "":
            lst_output.append(converter.underlined_str(self.__memory_mio_throttle.name()))
            super()._add_result_part_to_lst(self.__memory_mio_throttle.metrics(), 
                self.__memory_mio_throttle.metrics_description(), lst_output)
        if  self._collect_metrics and self.__memory_tex_throttle.metrics_str() != "":
            lst_output.append(converter.underlined_str(self.__memory_tex_throttle.name()))
            super()._add_result_part_to_lst(self.__memory_tex_throttle.metrics(), 
                self.__memory_tex_throttle.metrics_description(), lst_output)
        if self._collect_metrics and self._divergence.metrics_str() != "":
            lst_output.append(converter.underlined_str(self._divergence.name()))
            super()._add_result_part_to_lst(self._divergence.metrics(), 
                self._divergence.metrics_description(), lst_output)
        if self._collect_metrics and  self._retire.metrics_str() != "":
                lst_output.append(converter.underlined_str(self._retire.name()))
                super()._add_result_part_to_lst(self._retire.metrics(), 
                self._retire.metrics_description(), lst_output)
        if self._collect_metrics and self._extra_measure.metrics_str() != "":
            lst_output.append(converter.underlined_str(self._extra_measure.name()))
            super()._add_result_part_to_lst(self._extra_measure.metrics(), 
                self._extra_measure.metrics_description(), lst_output)
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

    def _set_memory_constant_memory_bound_mio_tex_throttle_results(self, results_launch : str):
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
        memory_mio_throttle_value_has_found: bool
        memory_mio_throttle_unit_has_found : bool
        memory_tex_throttle_value_has_found: bool
        memory_tex_throttle_unit_has_found : bool
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
                memory_constant_memory_bound_value_has_found = self.__memory_constant_memory_bound.set_metric_value(metric_name, 
                    metric_value)
                memory_constant_memory_bound_unit_has_found = self.__memory_constant_memory_bound.set_metric_unit(metric_name, 
                    metric_unit) 
                memory_mio_throttle_value_has_found = self.__memory_mio_throttle.set_metric_value(metric_name, 
                    metric_value)
                memory_mio_throttle_unit_has_found = self.__memory_mio_throttle.set_metric_unit(metric_name, 
                    metric_unit)
                memory_tex_throttle_value_has_found = self.__memory_tex_throttle.set_metric_value(metric_name, 
                    metric_value)
                memory_tex_throttle_unit_has_found = self.__memory_tex_throttle.set_metric_unit(metric_name, 
                    metric_unit)
                if (not (memory_constant_memory_bound_value_has_found or memory_mio_throttle_value_has_found or memory_tex_throttle_value_has_found) 
                    or not (memory_constant_memory_bound_unit_has_found or memory_mio_throttle_unit_has_found or memory_mio_throttle_unit_has_found)):
                    if not self._metricExists(metric_name):
                        raise MetricNotAsignedToPart(metric_name)
        pass

    def memory_mio_throttle_stall(self) -> float:
        """
        Returns percent of stalls due to BackEnd.MemoryBound.MemoryMioThrottle part.

        Returns:
            Float with percent of total stalls due to BackEnd.MemoryBound.MemoryMioThrottle part
        """

        return self._get_stalls_of_part(self.memory_mio_throttle().metrics())
        pass

    def memory_mio_throttle_stall_on_back(self) -> float:
        """ 
        Obtain the percentage of stalls due to BackEnd.MemoryBound.MemoryMioThrottle # repasar estos nombres en todo
        on the total BackEnd

        Returns:
            Float the percentage of stalls due to BackEnd.MemoryBound.MemoryMioThrottle
            on the total BackEnd
        """

        return (self.memory_mio_throttle_stall()/super().back_end_stall())*100.0

    def memory_mio_throttle_stall_on_memory_bound(self) -> float:
        """ 
        Obtain the percentage of stalls due to BackEnd.MemoryBound.MemoryMioThrottle
        on the total BackEnd.MemoryBound

        Returns:
            Float the percentage of stalls due to BackEnd.MemoryBound.MemoryMioThrottle
            on the total BackEnd.MemoryBound
        """

        return (self.memory_mio_throttle_stall()/super().back_memory_bound_stall())*100.0
        pass

    def memory_mio_throttle_percentage_ipc_degradation(self) -> float: # repasar nombres... Incluyen superior TODO
        """
        Find percentage of IPC degradation due to BackEnd.MemoryBound.MemoryMioThrottle part.

        Returns:
            Float with the percent of BackEnd.MemoryBound.MemoryMioThrottle's IPC degradation
        """

        return (((self._stall_ipc()*(self.memory_mio_throttle_stall()/100.0))/self.get_device_max_ipc())*100.0)
        pass
    
    def memory_tex_throttle_stall(self) -> float:
        """
        Returns percent of stalls due to BackEnd.MemoryBound.MemoryTexThrottle part.

        Returns:
            Float with percent of total stalls due to BackEnd.MemoryBound.MemoryTexThrottle part
        """

        return self._get_stalls_of_part(self.memory_tex_throttle().metrics())
        pass

    def memory_tex_throttle_stall_on_back(self) -> float:
        """ 
        Obtain the percentage of stalls due to BackEnd.MemoryBound.MemoryTexThrottle # repasar estos nombres en todo
        on the total BackEnd

        Returns:
            Float the percentage of stalls due to BackEnd.MemoryBound.MemoryTexThrottle
            on the total BackEnd
        """

        return (self.memory_tex_throttle_stall()/super().back_end_stall())*100.0

    def memory_tex_throttle_stall_on_memory_bound(self) -> float:
        """ 
        Obtain the percentage of stalls due to BackEnd.MemoryBound.MemoryTexThrottle
        on the total BackEnd.MemoryBound

        Returns:
            Float the percentage of stalls due to BackEnd.MemoryBound.MemoryTexThrottle
            on the total BackEnd.MemoryBound
        """

        return (self.memory_tex_throttle_stall()/super().back_memory_bound_stall())*100.0
        pass

    def memory_tex_throttle_percentage_ipc_degradation(self) -> float: # repasar nombres... Incluyen superior TODO
        """
        Find percentage of IPC degradation due to BackEnd.MemoryBound.MemoryTexThrottle part.

        Returns:
            Float with the percent of BackEnd.MemoryBound.MemoryTexThrottle's IPC degradation
        """

        return (((self._stall_ipc()*(self.memory_tex_throttle_stall()/100.0))/self.get_device_max_ipc())*100.0)
        pass

