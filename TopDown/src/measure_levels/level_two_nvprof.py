import re
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(1, parentdir) 
from measure_levels.level_one_nvprof import LevelOneNvprof
from measure_parts.back_core_bound import BackCoreBoundNvprof
from measure_parts.back_memory_bound import BackMemoryBoundNvprof
from measure_parts.front_band_width import FrontBandWidthNvprof
from measure_parts.front_dependency import FrontDependencyNvprof
from measure_parts.front_end import FrontEndNvprof
from measure_parts.back_end import BackEndNvprof
from measure_parts.divergence import DivergenceNvprof
from measure_parts.retire import RetireNvprof
from measure_parts.extra_measure import ExtraMeasureNvprof
from measure_levels.level_two import LevelTwo
from show_messages.message_format import MessageFormat
from errors.level_execution_errors import *

class LevelTwoNvprof(LevelTwo, LevelOneNvprof):
    """
    Class with level two of the execution based on NVPROF scan tool
    
    Atributes:
        _back_memory_bound      : BackMemoryBoundNvprof     ; backs' memory bound part
        _back_core_bound        : BackCoreBoundNvprof       ; backs' core bound part
        _front_band_width       : FrontBandWidthNvprof      ; front's bandwith part
        _front_dependency       : FrontDependencyNvprof     ; front's dependency part
    """

    def __init__(self, program : str, output_file : str, recoltect_metrics : bool, recolect_events : bool,
        front_end : FrontEndNvprof, back_end : BackEndNvprof, divergence : DivergenceNvprof, retire : RetireNvprof, 
        extra_measure : ExtraMeasureNvprof, front_band_width : FrontBandWidthNvprof, front_dependency : FrontDependencyNvprof, 
        back_core_bound : BackCoreBoundNvprof, back_memory_bound : BackMemoryBoundNvprof):
        
        self._back_core_bound : BackCoreBoundNvprof = back_core_bound
        self._back_memory_bound : BackMemoryBoundNvprof = back_memory_bound
        self._front_band_width : FrontBandWidthNvprof = front_band_width
        self._front_dependency : FrontDependencyNvprof = front_dependency
        super().__init__(program, output_file, recoltect_metrics, recolect_events, front_end, back_end, divergence, retire, extra_measure)
        pass

    def back_core_bound(self) -> BackCoreBoundNvprof:
        """
        Return CoreBound part of the execution.

        Returns:
            reference to CoreBound part of the execution
        """
        
        return self._back_core_bound
        pass

    def back_memory_bound(self) -> BackMemoryBoundNvprof:
        """
        Return MemoryBoundNvprof part of the execution.

        Returns:
            reference to MemoryBoundNvprof part of the execution
        """
        
        return self._back_memory_bound
        pass

    def front_band_width(self) -> FrontBandWidthNvprof:
        """
        Return FrontBandWidthNvprof part of the execution.

        Returns:
            reference to FrontBandWidthNvprof part of the execution
        """
        
        return self._front_band_width
        pass

    def front_dependency(self) -> FrontDependencyNvprof:
        """
        Return FrontDependencyNvprof part of the execution.

        Returns:
            reference to FrontDependencyNvprof part of the execution
        """
        
        return self._front_dependency
        pass

    def _generate_command(self) -> str:
        """ 
        Generate command of execution with NVIDIA scan tool.

        Returns:
            String with command to be executed
        """
        command : str = ("sudo $(which nvprof) --metrics " + self._front_end.metrics_str() + "," + self._back_end.metrics_str() + 
            "," + self._divergence.metrics_str() + "," + self._extra_measure.metrics_str() + "," + self._retire.metrics_str() + 
            "," + self._front_band_width.metrics_str() + "," + self._front_dependency.metrics_str() + "," + 
            self._back_core_bound.metrics_str() + "," + self._back_memory_bound.metrics_str() + "  --events " + 
            self._front_end.events_str() + "," + self._back_end.events_str() + "," + self._divergence.events_str() +  "," + 
            self._extra_measure.events_str() + "," + self._retire.events_str() + "," +  self._front_band_width.events_str() + 
            "," + self._front_dependency.events_str() + "," + self._back_core_bound.events_str() + "," + self._back_memory_bound.events_str() + 
            " --unified-memory-profiling off " + self._program)
        print(command)
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

        if not self._recolect_metrics and not self._recolect_events:
            return
        if (self._recolect_metrics and self._front_end.metrics_str() != "" or 
            self._recolect_events and self._front_end.events_str() != ""):
            lst_output.append(converter.underlined_str(self._front_end.name()))
        if self._recolect_metrics and self._front_end.metrics_str() != "":
            super()._add_result_part_to_lst(self._front_end.metrics(), 
                self._front_end.metrics_description(), lst_output, True)
        if self._recolect_events and self._front_end.events_str() != "":
                super()._add_result_part_to_lst(self._front_end.events(), 
                self._front_end.events_description(), "", lst_output, False)
        if (self._recolect_metrics and self._front_band_width.metrics_str() != "" or 
            self._recolect_events and self._front_band_width.events_str() != ""):
            lst_output.append(converter.underlined_str(self._front_band_width.name()))
        if  self._recolect_metrics and self._front_band_width.metrics_str() != "":
            super()._add_result_part_to_lst(self._front_band_width.metrics(), 
                self._front_band_width.metrics_description(), lst_output, True)
        if self._recolect_events and self._front_band_width.events_str() != "":
                super()._add_result_part_to_lst(self._front_band_width.events(), 
                self._front_band_width.events_description(), lst_output, False)
        if (self._recolect_metrics and self._front_dependency.metrics_str() != "" or 
            self._recolect_events and self._front_dependency.events_str() != ""):
            lst_output.append(converter.underlined_str(self._front_dependency.name()))
        if self._recolect_metrics and self._front_dependency.metrics_str() != "":
            super()._add_result_part_to_lst(self._front_dependency.metrics(), 
                self._front_dependency.metrics_description(), lst_output, True)
        if self._recolect_events and self._front_dependency.events_str() != "":
                super()._add_result_part_to_lst(self._front_dependency.events(), 
                self._front_dependency.events_description(), lst_output, False)
        if (self._recolect_metrics and self._back_end.metrics_str() != "" or 
            self._recolect_events and self._back_end.events_str() != ""):
            lst_output.append(converter.underlined_str(self._back_end.name()))
        if self._recolect_metrics and self._back_end.metrics_str() != "":
            super()._add_result_part_to_lst(self._back_end.metrics(), 
                self._back_end.metrics_description(), lst_output, True)        
        if self._recolect_events and self._back_end.events_str() != "":
                super()._add_result_part_to_lst(self._back_end.events(), 
                self._back_end.events_description(), lst_output, False)
        if (self._recolect_metrics and self._back_core_bound.metrics_str() != "" or 
            self._recolect_events and self._back_core_bound.events_str() != ""):
            lst_output.append(converter.underlined_str(self._back_core_bound.name()))
        if self._recolect_metrics and self._back_core_bound.metrics_str() != "":
            super()._add_result_part_to_lst(self._back_core_bound.metrics(), 
                self._back_core_bound.metrics_description(), lst_output, True)
        if self._recolect_events and self._back_core_bound.events_str() != "":
                super()._add_result_part_to_lst(self._back_core_bound.events(), 
                self._back_core_bound.events_description(), lst_output, False) 
        if (self._recolect_metrics and self._back_memory_bound.metrics_str() != "" or 
            self._recolect_events and self._back_memory_bound.events_str() != ""):
            lst_output.append(converter.underlined_str(self._back_memory_bound.name()))
        if self._recolect_metrics and self._back_memory_bound.metrics_str() != "":
            super()._add_result_part_to_lst(self._back_memory_bound.metrics(), 
                self._back_memory_bound.metrics_description(), lst_output, True)
        if self._recolect_events and self._back_memory_bound.events_str() != "":
                super()._add_result_part_to_lst(self._back_memory_bound.events(), 
                self._back_memory_bound.events_description(), lst_output, False)
        if (self._recolect_metrics and self._divergence.metrics_str() != "" or 
            self._recolect_events and self._divergence.events_str() != ""):
            lst_output.append(converter.underlined_str(self._divergence.name()))
        if self._recolect_metrics and self._divergence.metrics_str() != "":
            super()._add_result_part_to_lst(self._divergence.metrics(), 
                self._divergence.metrics_description(), lst_output, True)
        if self._recolect_events and self._divergence.events_str() != "":
                super()._add_result_part_to_lst(self._divergence.events(), 
                self._divergence.events_description(),lst_output, False)
        if (self._recolect_metrics and self._retire.metrics_str() != "" or 
            self._recolect_events and self._retire.events_str() != ""):
            lst_output.append(converter.underlined_str(self._retire.name()))  
        if self._recolect_metrics and  self._retire.metrics_str() != "":
                super()._add_result_part_to_lst(self._retire.metrics(), 
                self._retire.metrics_description(), lst_output, True)
        if self._recolect_events and self._retire.events_str() != "":
                super()._add_result_part_to_lst(self._retire.events(), 
                self._retire.events_description(), lst_output, False)
        if (self._recolect_metrics and self._extra_measure.metrics_str() != "" or 
            self._recolect_events and self._extra_measure.events_str() != ""):
            lst_output.append(converter.underlined_str(self._extra_measure.name()))
        if self._recolect_metrics and self._extra_measure.metrics_str() != "":
            super()._add_result_part_to_lst(self._extra_measure.metrics(), 
                self._extra_measure.metrics_description(), lst_output, True)
        if self._recolect_events and self._extra_measure.events_str() != "":
                super()._add_result_part_to_lst(self._extra_measure.events(), 
                self._extra_measure.events_description(), lst_output, False)
        lst_output.append("\n")
        pass
    
    def _set_memory_core_bandwith_dependency_results(self, results_launch : str):
        """
        Set results of the level two part (that are not level one).
        
        Params:
            results_launch : str   ; results generated by NVIDIA scan tool.
        Raises:
             MetricNotAsignedToPart ; raised if some metric is found don't assigned 
                                      to any measure part
 
             EventNotAsignedToPart ; raised if some event is found don't assigned 
                                      to any measure part
        """

        event_name : str
        event_total_value : str
        metric_name : str
        metric_description : str
        metric_avg_value : str
        has_read_all_events : bool = False
        back_core_bound_value_has_found : bool 
        back_core_bound_description_has_found : bool
        back_memory_bound_value_has_found : bool 
        back_memory_bound_description_has_found : bool
        front_band_width_value_has_found : bool 
        front_band_width_description_has_found : bool
        front_dependency_value_has_found : bool 
        front_dependency_description_has_found : bool
        for line in results_launch.splitlines():
            line = re.sub(' +', ' ', line) # delete more than one spaces and put only one
            list_words = line.split(" ")
            if not has_read_all_events:
                # Check if it's line of interest:
                # ['', 'X', 'event_name','Min', 'Max', 'Avg', 'Total']
                if len(list_words) > 1: 
                    if list_words[1] == "Metric": # check end events
                        has_read_all_events = True
                    elif list_words[0] == '' and list_words[len(list_words) - 1][0].isnumeric():
                        event_name = list_words[2]
                        event_total_value = list_words[len(list_words) - 1]     
                        back_core_bound_value_has_found = self._back_core_bound.set_event_value(event_name, event_total_value) #TODO atributo
                        #back_core_bound_description_has_found = self._back_core_bound.set_event_description(event_name, metric_description)
                        back_memory_bound_value_has_found = self._back_memory_bound.set_event_value(event_name, event_total_value)
                        #back_memory_bound_description_has_found = self._back_memory_bound.set_event_description(event_name, metric_description)
                        front_band_width_value_has_found = self._front_band_width.set_event_value(event_name, event_total_value)
                        #front_band_width_description_has_found = self._front_band_width.set_event_description(event_name, metric_description)
                        front_dependency_value_has_found = self._front_dependency.set_event_value(event_name, event_total_value)
                        #front_dependency_description_has_found = self._front_dependency.set_event_description(event_name, metric_description)
                        if not (back_core_bound_value_has_found or back_memory_bound_value_has_found or front_band_width_value_has_found
                            or front_dependency_value_has_found): #or 
                            #not(back_core_bound_description_has_found or back_memory_bound_description_has_found or front_band_width_description_has_found
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
                    back_core_bound_value_has_found = self._back_core_bound.set_metric_value(metric_name, metric_avg_value)
                    back_core_bound_description_has_found = self._back_core_bound.set_metric_description(metric_name, metric_description)
                    back_memory_bound_value_has_found = self._back_memory_bound.set_metric_value(metric_name, metric_avg_value)
                    back_memory_bound_description_has_found = self._back_memory_bound.set_metric_description(metric_name, metric_description)
                    front_band_width_value_has_found = self._front_band_width.set_metric_value(metric_name, metric_avg_value)
                    front_band_width_description_has_found = self._front_band_width.set_metric_description(metric_name, metric_description)  
                    front_dependency_value_has_found = self._front_dependency.set_metric_value(metric_name, metric_avg_value)
                    front_dependency_description_has_found = self._front_dependency.set_metric_description(metric_name, metric_description)         
                    if (not (back_core_bound_value_has_found or back_memory_bound_value_has_found or front_band_width_value_has_found
                        or front_dependency_value_has_found) or not(back_core_bound_description_has_found or back_memory_bound_description_has_found 
                        or front_band_width_description_has_found or front_dependency_description_has_found)): 
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
        is_defined_front_end_value : str = super().front_end().get_metric_value(metric_name)
        is_defined_front_end_description : str = super().front_end().get_metric_description(metric_name)

        is_defined_back_end_value : str = super().back_end().get_metric_value(metric_name)
        is_defined_back_end_description : str = super().back_end().get_metric_description(metric_name)

        is_defined_divergence_value : str = super().divergence().get_metric_value(metric_name)
        is_defined_divergence_description : str = super().divergence().get_metric_description(metric_name)

        is_defined_retire_value : str = super().retire().get_metric_value(metric_name)
        is_defined_retire_description : str = super().retire().get_metric_description(metric_name)

        is_defined_extra_measure_value : str = super().extra_measure().get_metric_value(metric_name)
        is_defined_extra_measure_description : str = super().extra_measure().get_metric_description(metric_name)

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
        """ #TODO se podria poner este metodo (en version abstracta) en otra clase aparte, para que el LevelTwoNsight compruebe si existe
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

