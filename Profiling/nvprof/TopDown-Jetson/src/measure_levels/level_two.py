"""
Class that represents the level two of the execution

@author:    Alvaro Saiz (UC)
@date:      Jan-2021
@version:   1.0
"""

import re
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from errors.level_execution_errors import *
from measure_parts.core_bound import CoreBound
from measure_parts.memory_bound import MemoryBound
from errors.level_execution_errors import *
from measure_parts.front_band_width import FrontBandWidth
from measure_parts.front_dependency import FrontDependency
from measure_levels.level_one import LevelOne
from show_messages.message_format import MessageFormat


class LevelTwo(LevelOne):
    """
    Class with level two of the execution.
    
    Atributes:
        _back_memory_bound : MemoryBound       ; memory bound part
        _back_core_bound        : CoreBound         ; core bound part
        _front_band_width  : FrontBandWidth    ; front's bandwith part
        _front_dependency  : FrontDepdendency  ; front's dependency part
    """

    def __init__(self, program : str, output_file : str, recoltect_metrics : bool, recolect_events : bool, is_tesla_device : bool):
        
        self._back_core_bound : CoreBound = CoreBound()
        self._back_memory_bound : MemoryBound = MemoryBound()
        self._front_band_width : FrontBandWidth = FrontBandWidth()
        self._front_dependency : FrontDependency = FrontDependency()
        super().__init__(program, output_file, recoltect_metrics, recolect_events, is_tesla_device)
        pass

    def back_core_bound(self) -> CoreBound:
        """
        Return CoreBound part of the execution.

        Returns:
            reference to CoreBound part of the execution
        """
        
        return self._back_core_bound
        pass

    def back_memory_bound(self) -> MemoryBound:
        """
        Return MemoryBound part of the execution.

        Returns:
            reference to MemoryBound part of the execution
        """
        
        return self._back_memory_bound
        pass

    def front_band_width(self) -> FrontBandWidth:
        """
        Return FrontBandWidth part of the execution.

        Returns:
            reference to FrontBandWidth part of the execution
        """
        
        return self._front_band_width
        pass

    def front_dependency(self) -> FrontDependency:
        """
        Return FrontDependency part of the execution.

        Returns:
            reference to FrontDependency part of the execution
        """
        
        return self._front_dependency
        pass

    def _generate_command(self) -> str:
        """ 
        Generate command of execution with NVIDIA scan tool.

        Returns:
            String with command to be executed
        """

        command : str = ("sudo $(which nvprof) --metrics " + self._front_end.metrics_str() + 
            "," + self._back_end.metrics_str() + "," + self._divergence.metrics_str() + "," + self._extra_measure.metrics_str()
            + "," + self._retire.metrics_str() + "," + self._back_core_bound.metrics_str() + "," + self._back_memory_bound.metrics_str() + 
            "  --events " + self._front_end.events_str() + "," + self._back_end.events_str() + "," + self._divergence.events_str() +  
            "," + self._extra_measure.events_str() + "," + self._retire.events_str() + "," + self._back_core_bound.events_str() + 
            "," + self._back_memory_bound.events_str() +" --unified-memory-profiling off --profile-from-start off " + self._program)
        return command
        pass

    def __metricExists(self, metric_name : str) -> bool:
        """
        Check if metric exists in some part of the execution (MemoryBound, CoreBound...). 

        Params:
            metric_name  : str   ; name of the metric to be checked

        Returns:
            True if metric is defined in some part of the execution (MemoryBound, CoreBound...)
            or false in other case
        """

        # revisar si las superiores (core bound sobre back end) tiene que estar definida
        is_defined_front_end_value : str = self.front_end().get_metric_value(metric_name)
        is_defined_front_end_description : str = self.front_end().get_metric_description(metric_name)

        is_defined_back_end_value : str = self.back_end().get_metric_value(metric_name)
        is_defined_back_end_description : str = self.back_end().get_metric_description(metric_name)

        is_defined_divergence_value : str = self.divergence().get_metric_value(metric_name)
        is_defined_divergence_description : str = self.divergence().get_metric_description(metric_name)

        is_defined_retire_value : str = self.retire().get_metric_value(metric_name)
        is_defined_retire_description : str = self.retire().get_metric_description(metric_name)

        is_defined_extra_measure_value : str = self.extra_measure().get_metric_value(metric_name)
        is_defined_extra_measure_description : str = self.extra_measure().get_metric_description(metric_name)

        # comprobar el "if not"
        if not ((is_defined_front_end_value is None and is_defined_front_end_description is None) 
            or (is_defined_back_end_value is None and is_defined_back_end_description is None)
            or (is_defined_divergence_value is None and is_defined_divergence_description is None) 
            or (is_defined_retire_value is None or is_defined_retire_description is None)
            or (is_defined_extra_measure_value is None and is_defined_extra_measure_description is None)):
            return False
        return True
        pass

    def __eventExists(self, event_name : str) -> bool:
        """
        Check if event exists in some part of the execution (MemoryBound, CoreBound...). 

        Params:
            event_name  : str   ; name of the event to be checked

        Returns:
            True if event is defined in some part of the execution (MemoryBound, CoreBound...)
            or false in other case
        """

        # revisar si las superiores (core bound sobre back end pe) tiene que estar definida
        is_defined_front_end_value : str = self.front_end().get_metric_value(event_name)
        is_defined_front_end_description : str = self.front_end().get_metric_description(event_name)

        is_defined_back_end_value : str = self.back_end().get_metric_value(event_name)
        is_defined_back_end_description : str = self.back_end().get_metric_description(event_name)

        is_defined_divergence_value : str = self.divergence().get_metric_value(event_name)
        is_defined_divergence_description : str = self.divergence().get_metric_description(event_name)

        is_defined_retire_value : str = self.retire().get_metric_value(event_name)
        is_defined_retire_description : str = self.retire().get_metric_description(event_name)

        is_defined_extra_measure_value : str = self.extra_measure().get_metric_value(event_name)
        is_defined_extra_measure_description : str = self.extra_measure().get_metric_description(event_name)

        # revisar IF, puede ser (1 y 0) y (1 y 1). EL primero no puede darse pero cumpliria
        if not ((is_defined_front_end_value is None and is_defined_front_end_description is None) 
            or (is_defined_back_end_value is None and is_defined_back_end_description is None)
            or (is_defined_divergence_value is None and is_defined_divergence_description is None) 
            or (is_defined_retire_value is None or is_defined_retire_description is None)
            or (is_defined_extra_measure_value is None and is_defined_extra_measure_description is None)):
            return False
        return True
        pass

    def _get_results(self, lst_output : list[str]):
        """ 
        Get results of the different parts.

        Parameters:
            lst_output              : list[str]     ; OUTPUT list with results
        """

        # revisar en unos usa atributo y en otros la llamada al metodo
        #  Keep Results
        converter : MessageFormat = MessageFormat()

        if not self._recolect_metrics and not self._recolect_events:
            return

        if (self._recolect_metrics and self._front_end.metrics_str() != "" or 
            self._recolect_events and self._front_end.events_str() != ""):
            lst_output.append(converter.underlined_str(self._front_end.name()))
        if self._recolect_metrics and self._front_end.metrics_str() != "":
            self._add_result_part_to_lst(self._front_end.metrics(), 
                self._front_end.metrics_description(), lst_output, True)
        if self._recolect_events and self._front_end.events_str() != "":
                self._add_result_part_to_lst(self._front_end.events(), 
                self._front_end.events_description(), "", lst_output, False)
        if (self._recolect_metrics and self._front_band_width.metrics_str() != "" or 
            self._recolect_events and self._front_band_width.events_str() != ""):
            lst_output.append(converter.underlined_str(self._front_band_width.name()))
        if  self._recolect_metrics and self._front_band_width.metrics_str() != "":
            self._add_result_part_to_lst(self._front_band_width.metrics(), 
                self._front_band_width.metrics_description(), lst_output, True)
        if self._recolect_events and self._front_band_width.events_str() != "":
                self._add_result_part_to_lst(self._front_band_width.events(), 
                self._front_band_width.events_description(), lst_output, False)
        if (self._recolect_metrics and self._front_dependency.metrics_str() != "" or 
            self._recolect_events and self._front_dependency.events_str() != ""):
            lst_output.append(converter.underlined_str(self._front_dependency.name()))
        if self._recolect_metrics and self._front_dependency.metrics_str() != "":
            self._add_result_part_to_lst(self._front_dependency.metrics(), 
                self._front_dependency.metrics_description(), lst_output, True)
        if self._recolect_events and self._front_dependency.events_str() != "":
                self._add_result_part_to_lst(self._front_dependency.events(), 
                self._front_dependency.events_description(), lst_output, False)
        if (self._recolect_metrics and self._back_end.metrics_str() != "" or 
            self._recolect_events and self._back_end.events_str() != ""):
            lst_output.append(converter.underlined_str(self._back_end.name()))
        if self._recolect_metrics and self._back_end.metrics_str() != "":
            self._add_result_part_to_lst(self._back_end.metrics(), 
                self._back_end.metrics_description(), lst_output, True)
        if self._recolect_events and self._back_end.events_str() != "":
                self._add_result_part_to_lst(self._back_end.events(), 
                self._back_end.events_description(), lst_output, False)
        if (self._recolect_metrics and self._back_core_bound.metrics_str() != "" or 
            self._recolect_events and self._back_core_bound.events_str() != ""):
            lst_output.append(converter.underlined_str(self._back_core_bound.name()))
        if self._recolect_metrics and self._back_core_bound.metrics_str() != "":
            self._add_result_part_to_lst(self._back_core_bound.metrics(), 
                self._back_core_bound.metrics_description(), lst_output, True)
        if self._recolect_events and self._back_core_bound.events_str() != "":
                self._add_result_part_to_lst(self._back_core_bound.events(), 
                self._back_core_bound.events_description(), lst_output, False) 
        if (self._recolect_metrics and self._back_memory_bound.metrics_str() != "" or 
            self._recolect_events and self._back_memory_bound.events_str() != ""):
            lst_output.append(converter.underlined_str(self._back_memory_bound.name()))
        if self._recolect_metrics and self._back_memory_bound.metrics_str() != "":
            self._add_result_part_to_lst(self._back_memory_bound.metrics(), 
                self._back_memory_bound.metrics_description(), lst_output, True)
        if self._recolect_events and self._back_memory_bound.events_str() != "":
                self._add_result_part_to_lst(self._back_memory_bound.events(), 
                self._back_memory_bound.events_description(), lst_output, False)
        if (self._recolect_metrics and self._divergence.metrics_str() != "" or 
            self._recolect_events and self._divergence.events_str() != ""):
            lst_output.append(converter.underlined_str(self._divergence.name()))
        if self._recolect_metrics and self._divergence.metrics_str() != "":
            self._add_result_part_to_lst(self._divergence.metrics(), 
                self._divergence.metrics_description(), lst_output, True)
        if self._recolect_events and self._divergence.events_str() != "":
                self._add_result_part_to_lst(self._divergence.events(), 
                self._divergence.events_description(),lst_output, False)
        if (self._recolect_metrics and self._retire.metrics_str() != "" or 
            self._recolect_events and self._retire.events_str() != ""):
            lst_output.append(converter.underlined_str(self._retire.name()))
        if self._recolect_metrics and  self._retire.metrics_str() != "":
                self._add_result_part_to_lst(self._retire.metrics(), 
                self._retire.metrics_description(), lst_output, True)
        if self._recolect_events and self._retire.events_str() != "":
                self._add_result_part_to_lst(self._retire.events(), 
                self._retire.events_description(), lst_output, False)
        if (self._recolect_metrics and self._extra_measure.metrics_str() != "" or 
            self._recolect_events and self._extra_measure.events_str() != ""):
            lst_output.append(converter.underlined_str(self._extra_measure.name()))
        if self._recolect_metrics and self._extra_measure.metrics_str() != "":
            self._add_result_part_to_lst(self._extra_measure.metrics(), 
                self._extra_measure.metrics_description(), lst_output, True)
        if self._recolect_events and self._extra_measure.events_str() != "":
                self._add_result_part_to_lst(self._extra_measure.events(), 
                self._extra_measure.events_description(), lst_output, False)
        lst_output.append("\n")
        pass

    def _set_memory_core_bandwith_dependency_results(self, results_launch : str):
        """
        Set results of the level two part (that are not level one).
        
        Params:
            results_launch : str   ; results generated by NVIDIA scan tool.
        """

        has_read_all_events : bool = False
        core_bound_value_has_found : bool 
        core_bound_description_has_found : bool
        memory_bound_value_has_found : bool 
        memory_bound_description_has_found : bool
        front_band_width_value_has_found : bool 
        front_band_width_description_has_found : bool
        front_dependency_value_has_found : bool 
        front_dependency_description_has_found : bool
        for line in results_launch.splitlines():
            line = re.sub(' +', ' ', line) # delete more than one spaces and put only one
            list_words = line.split(" ")
            if not has_read_all_events:
                # Check if it's line of interest:
                # ['', 'X', 'event_name','Min', 'Max', 'Avg', 'Total'] event_name is str. Rest: numbers (less '', it's an space)
                if len(list_words) > 1: 
                    if list_words[1] == "Metric": # check end events
                        has_read_all_events = True
                    elif list_words[0] == '' and list_words[len(list_words) - 1][0].isnumeric():
                        event_name = list_words[2]
                        event_total_value = list_words[len(list_words) - 1]     
                        core_bound_value_has_found = self._back_core_bound.set_event_value(event_name, event_total_value)
                        #core_bound_description_has_found = self._back_core_bound.set_event_description(event_name, metric_description)
                        memory_bound_value_has_found = self._back_memory_bound.set_event_value(event_name, event_total_value)
                        #memory_bound_description_has_found = self._back_memory_bound.set_event_description(event_name, metric_description)
                        front_band_width_value_has_found = self._front_band_width.set_event_value(event_name, event_total_value)
                        #front_band_width_description_has_found = self._front_band_width.set_event_description(event_name, metric_description)
                        front_dependency_value_has_found = self._front_dependency.set_event_value(event_name, event_total_value)
                        #front_dependency_description_has_found = self._front_dependency.set_event_description(event_name, metric_description)
                        if not (core_bound_value_has_found or memory_bound_value_has_found or front_band_width_value_has_found
                            or front_dependency_value_has_found): #or 
                            #not(core_bound_description_has_found or memory_bound_description_has_found or front_band_width_description_has_found
                            # or front_dependency_description_has_found)): 
                            if not self.__eventExists(event_name):
                                raise EventNotAsignedToPart(event_name)
            else: # metrics
                # Check if it's line of interest:
                # ['', 'X', 'NAME_COUNTER', ... , 'Min', 'Max', 'Avg' (Y%)] where X (int number), Y (int/float number)
                if len(list_words) > 1 and list_words[0] == '' and list_words[len(list_words) - 1][0].isnumeric():
                    metric_name = list_words[2]
                    metric_description = ""
                    for i in range(3, len(list_words) - 3):
                        metric_description += list_words[i] + " "     
                    metric_avg_value = list_words[len(list_words) - 1]
                    #metric_max_value = list_words[len(list_words) - 2]
                    #metric_min_value = list_words[len(list_words) - 3]
                    #if metric_avg_value != metric_max_value or metric_avg_value != metric_min_value:
                        # Do Something. NOT USED
                    
                    core_bound_value_has_found = self._back_core_bound.set_metric_value(metric_name, metric_avg_value)
                    core_bound_description_has_found = self._back_core_bound.set_metric_description(metric_name, metric_description)
                    memory_bound_value_has_found = self._back_memory_bound.set_metric_value(metric_name, metric_avg_value)
                    memory_bound_description_has_found = self._back_memory_bound.set_metric_description(metric_name, metric_description)
                    front_band_width_value_has_found = self._front_band_width.set_metric_value(metric_name, metric_avg_value)
                    front_band_width_description_has_found = self._front_band_width.set_metric_description(metric_name, metric_description)  
                    front_dependency_value_has_found = self._front_dependency.set_metric_value(metric_name, metric_avg_value)
                    front_dependency_description_has_found = self._front_dependency.set_metric_description(metric_name, metric_description)         
                    if (not (core_bound_value_has_found or memory_bound_value_has_found or front_band_width_value_has_found
                        or front_dependency_value_has_found) or not(core_bound_description_has_found or memory_bound_description_has_found 
                        or front_band_width_description_has_found or front_dependency_description_has_found)): 
                        if not self.__metricExists(metric_name):
                            raise MetricNotAsignedToPart(metric_name)

    def run(self, lst_output : list[str]):
        """Run execution."""
        
        # compute results
        output_command : str = super()._launch(self._generate_command())
        super()._set_front_back_divergence_retire_results(output_command) # level one results
        self._set_memory_core_bandwith_dependency_results(output_command)
        self._get_results(lst_output)
        pass
    
    def back_core_bound_percentage_ipc_degradation(self) -> float:
        """
        Find percentage of IPC degradation due to BackEnd.Core_Bound part.

        Returns:
            Float with the percent of BackEnd.Core_Bound's IPC degradation
        """
        
        return (((self._stall_ipc()*(self.get_back_core_bound_stall()/100.0))/self.get_device_max_ipc())*100.0)
        pass

    def back_memory_bound_percentage_ipc_degradation(self) -> float:
        """
        Find percentage of IPC degradation due to BackEnd.Memory_Bound part.

        Returns:
            Float with the percent of BackEnd.Memory_Bound's IPC degradation
        """

        return (((self._stall_ipc()*(self.get_back_memory_bound_stall()/100.0))/self.get_device_max_ipc())*100.0)
        pass

    def front_band_width_percentage_ipc_degradation(self) -> float:
        """
        Find percentage of IPC degradation due to FrontEnd.BandWidth part.

        Returns:
            Float with the percent of FrontEnd.BandWidth's IPC degradation
        """

        return (((self._stall_ipc()*(self.get_front_band_width_stall()/100.0))/self.get_device_max_ipc())*100.0)
        pass

    def front_dependency_percentage_ipc_degradation(self) -> float:
        """
        Find percentage of IPC degradation due to FrontEnd.Dependency part.

        Returns:
            Float with the percent of FrontEnd.Dependency's IPC degradation
        """

        return (((self._stall_ipc()*(self.get_front_dependency_stall()/100.0))/self.get_device_max_ipc())*100.0)
        pass

    def get_back_memory_bound_stall(self) -> float:
        """
        Returns percent of stalls due to BackEnd.Memory_Bound part.

        Returns:
            Float with percent of total stalls due to BackEnd.Memory_Bound
        """

        return self._get_stalls_of_part(self._back_memory_bound.metrics())
        pass

    def get_back_core_bound_stall(self) -> float:
        """
        Returns percent of stalls due to BackEnd.Core_Bound part.

        Returns:
            Float with percent of total stalls due to BackEnd.Core_Bound
        """

        return self._get_stalls_of_part(self._back_core_bound.metrics())
        pass

    def get_front_band_width_stall(self) -> float:
        """
        Returns percent of stalls due to FrontEnd.Band_width part.

        Returns:
            Float with percent of total stalls due to FrontEnd.Band_width part
        """

        return self._get_stalls_of_part(self._front_band_width.metrics())
        pass

    def get_front_dependency_stall(self) -> float:
        """
        Returns percent of stalls due to FrontEnd.Dependency part.

        Returns:
            Float with percent of total stalls due to FrontEnd.Dependency part
        """

        return self._get_stalls_of_part(self._front_dependency.metrics())
        pass

    def get_back_memory_bound_stall_on_back(self) -> float:
        """ 
        Obtain the percentage of stalls due to BackEnd.Memory_Bound
        on the total BackEnd

        Returns:
            Float the percentage of stalls due to BackEnd.Memory_Bound
            on the total BackEnd
        """

        return (self.get_back_memory_bound_stall()/super().get_back_end_stall())*100.0 

    def get_back_core_bound_stall_on_back(self) -> float:
        """ 
        Obtain the percentage of stalls due to BackEnd.Core_Bound
        on the total BackEnd

        Returns:
            Float the percentage of stalls due to BackEnd.Core_Bound
        on the total BackEnd
        """

        return (self.get_back_core_bound_stall()/super().get_back_end_stall())*100.0 

    def get_front_band_width_stall_on_front(self) -> float:
        """ 
        Obtain the percentage of stalls due to FrontEnd.Band_width
        on the total FrontEnd

        Returns:
            Float the percentage of stalls due to FrontEnd.Band_width
            on the total FrontEnd
        """

        return (self.get_front_band_width_stall()/super().get_front_end_stall())*100.0 
        pass

    def get_front_dependency_stall_on_front(self) -> float:
        """ 
        Obtain the percentage of stalls due to FrontEnd.Dependency
        on the total FrontEnd

        Returns:
            Float the percentage of stalls due to FrontEnd.Dependency
            on the total FrontEnd
        """

        return (self.get_front_dependency_stall()/super().get_front_end_stall())*100.0 
        pass