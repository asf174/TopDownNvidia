from abc import ABC, abstractmethod # abstract class
import os, sys, inspect, re
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from measure_levels.level_one import LevelOne 
from measure_levels.level_execution_nvprof import LevelExecutionNvprof
from measure_levels.level_execution import LevelExecution 
from measure_parts.front_end import FrontEndNvprof
from measure_parts.back_end import BackEndNvprof
from measure_parts.divergence import DivergenceNvprof
from measure_parts.retire import RetireNvprof
from show_messages.message_format import MessageFormat
from parameters.level_execution_params import LevelExecutionParameters
from graph.pie_chart import PieChart

class LevelOneNvprof(LevelOne, LevelExecutionNvprof):


    """ 
    Class thath represents the level one of the execution with nvprof scan tool.

    Attributes:
        _front_end      : FrontEnd      ; FrontEnd part of the execution
        _back_end       : BackEnd       ; BackEnd part of the execution
        _divergence     : Divergence    ; Divergence part of the execution
        _retire         : Retire        ; Retire part of the execution
    """

    def __init__(self, program : str, output_file : str, recoltect_metrics : bool, recolect_events : bool):
        self._front_end : FrontEndNvprof = FrontEndNvprof()
        self._back_end  : BackEndNvprof = BackEndNvprof()
        self._divergence : DivergenceNvprof = DivergenceNvprof()
        self._retire : RetireNvprof = RetireNvprof()
        super().__init__(program, output_file, recoltect_metrics, recolect_events)
        pass

    def retire_ipc(self) -> float:
        """
        Get "RETIRE" IPC of execution.

        Raises:
            RetireIpcMetricNotDefined ; raised if retire IPC cannot be obtanied because it was not 
            computed by the NVIDIA scan tool.
        """

        return super()._ret_ipc(LevelExecutionParameters.C_WARP_EXECUTION_EFFICIENCY_METRIC_NAME_NVPROF)
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

    def run(self, lst_output : list):
        """Run execution."""
        
        output_command : str = super()._launch(self._generate_command())
        self._set_front_back_divergence_retire_results(output_command)
        self._get_results(lst_output)
        self.printGraph()
        pass

    def _set_front_back_divergence_retire_results(self, results_launch : str):
        """ Get Results from FrontEnd, BanckEnd, Divergence and Retire parts.
        
        Params:
            results_launch  : str   ; results generated by NVIDIA scan tool
            
        Raises:
            EventNotAsignedToPart       ; raised when an event has not been assigned to any analysis part 
            MetricNotAsignedToPart      ; raised when a metric has not been assigned to any analysis part
        """

        event_name : str
        event_total_value : str 
        metric_name : str
        metric_description : str = ""
        metric_avg_value : str 
        #metric_max_value : str 
        #metric_min_value : str
        has_read_all_events : bool = False
        line : str
        i : int
        list_words : list
        front_end_value_has_found : bool
        frond_end_description_has_found : bool
        back_end_value_has_found : bool
        back_end_description_has_found : bool
        divergence_value_has_found : bool
        divergence_description_has_found : bool
        extra_measure_value_has_found : bool
        extra_measure_description_has_found : bool
        retire_value_has_found : bool 
        retire_description_has_found : bool
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
                        front_end_value_has_found = self._front_end.set_event_value(event_name, event_total_value)
                        #frond_end_description_has_found = front_end.set_event_description(event_name, metric_description)
                        back_end_value_has_found = self._back_end.set_event_value(event_name, event_total_value)
                        #back_end_description_has_found = back_end.set_event_description(event_name, metric_description)
                        divergence_value_has_found = self._divergence.set_event_value(event_name, event_total_value)
                        #divergence_description_has_found = divergence.set_event_description(event_name, metric_description)
                        extra_measure_value_has_found = self._extra_measure.set_event_value(event_name, event_total_value)
                        #extra_measure_description_has_found = extra_measure.set_event_description(event_name, metric_description)
                        retire_value_has_found = self._retire.set_event_value(event_name, event_total_value)
                        #retire_description_has_found = extra_measure.set_event_description(event_name, metric_description)
                        if (not (front_end_value_has_found or back_end_value_has_found or divergence_value_has_found or 
                            extra_measure_value_has_found or retire_value_has_found)): #or 
                            #not(frond_end_description_has_found or back_end_description_has_found 
                            #or divergence_description_has_found or extra_measure_description_has_found)):
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
                    front_end_value_has_found = self._front_end.set_metric_value(metric_name, metric_avg_value)
                    frond_end_description_has_found = self._front_end.set_metric_description(metric_name, metric_description)
                    back_end_value_has_found = self._back_end.set_metric_value(metric_name, metric_avg_value)
                    back_end_description_has_found = self._back_end.set_metric_description(metric_name, metric_description)
                    divergence_value_has_found = self._divergence.set_metric_value(metric_name, metric_avg_value)
                    divergence_description_has_found = self._divergence.set_metric_description(metric_name, metric_description)
                    extra_measure_value_has_found = self._extra_measure.set_metric_value(metric_name, metric_avg_value)
                    extra_measure_description_has_found = self._extra_measure.set_metric_description(metric_name, metric_description)
                    retire_value_has_found = self._retire.set_metric_value(metric_name, metric_avg_value)
                    retire_description_has_found = self._retire.set_metric_description(metric_name, metric_description)
                    if (not (front_end_value_has_found or back_end_value_has_found or divergence_value_has_found or 
                        extra_measure_value_has_found or retire_value_has_found) or 
                        not(frond_end_description_has_found or back_end_description_has_found or divergence_description_has_found 
                        or extra_measure_description_has_found or retire_description_has_found)):
                        raise MetricNotAsignedToPart(metric_name)
        pass

    def front_end(self) -> FrontEndNvprof:
        """
        Return FrontEndNvprof part of the execution.

        Returns:
            reference to FrontEndNvprof part of the execution
        """

        return self._front_end
        pass
    
    def back_end(self) -> BackEndNvprof:
        """
        Return BackEndNvprof part of the execution.

        Returns:
            reference to BackEndNvprof part of the execution
        """

        return self._back_end
        pass

    def divergence(self) -> DivergenceNvprof:
        """
        Return DivergenceNvprof part of the execution.

        Returns:
            reference to DivergenceNvprof part of the execution
        """

        return self._divergence
        pass

    def retire(self) -> RetireNvprof:
        """
        Return RetireNvprof part of the execution.

        Returns:
            reference to RetireNvprof part of the execution
        """

        return self._retire
        pass
    
    def _divergence_ipc_degradation(self) -> float:
        """
        Find IPC degradation due to Divergence part

        Returns:
            Float with theDivergence's IPC degradation

        """

        return super()._diver_ipc_degradation(LevelExecutionParameters.C_WARP_EXECUTION_EFFICIENCY_METRIC_NAME_NVPROF, LevelExecutionParameters.C_ISSUE_IPC_METRIC_NAME_NVPROF)
        pass

    def _get_results(self, lst_output : list):
        """
        Get results of the different parts.

        Parameters:
            lst_output              : list     ; OUTPUT list with results
        """

        converter : MessageFormat = MessageFormat()
        #  Keep Results
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
        if (self._recolect_metrics and self._back_end.metrics_str() != "" or 
            self._recolect_events and self._back_end.events_str() != ""):
            lst_output.append(converter.underlined_str(self._back_end.name()))
        if self._recolect_metrics and self._back_end.metrics_str() != "":
            super()._add_result_part_to_lst(self._back_end.metrics(), 
                self._back_end.metrics_description(), lst_output, True)
        if self._recolect_events and self._back_end.events_str() != "":
                super()._add_result_part_to_lst(self._back_end.events(), 
                self._back_end.events_description(), lst_output, False)
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

    def ipc(self) -> float:
        """
        Get IPC of execution.

        Returns:
            float with the IPC
        """

        return super()._get_ipc(LevelExecutionParameters.C_IPC_METRIC_NAME_NVPROF)
        pass

    def printGraph(self):
        """
        Print graph to show results."""

        graph : PieChart = PieChart() # pie chart graph
        labels : list = [self._front_end.name(), self._back_end.name(), self._divergence.name(), self._retire.name()]
        sizes : list = [super().front_end_percentage_ipc_degradation(), super().back_end_percentage_ipc_degradation(), super().divergence_percentage_ipc_degradation(), super().retire_ipc()]
        explode : list = [0,0,0,0]
        graph.print(labels, sizes, explode)
        pass
