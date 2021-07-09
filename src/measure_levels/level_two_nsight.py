import re
from measure_levels.level_two import LevelTwo
from measure_levels.level_one_nsight import LevelOneNsight
from measure_parts.back_core_bound import BackCoreBoundNsight
from measure_parts.back_memory_bound import BackMemoryBoundNsight
from measure_parts.front_band_width import FrontBandWidthNsight
from measure_parts.front_dependency import FrontDependencyNsight
from measure_parts.divergence_replay import DivergenceReplayNsight
from measure_parts.divergence_branch import DivergenceBranchNsight
from measure_parts.front_end import FrontEndNsight
from measure_parts.back_end import BackEndNsight
from measure_parts.divergence import DivergenceNsight
from measure_parts.retire import RetireNsight
from measure_parts.extra_measure import ExtraMeasureNsight
from show_messages.message_format import MessageFormat
from errors.level_execution_errors import *
from parameters.divergence_replay_params import DivergenceReplayParameters
from parameters.divergence_branch_params import DivergenceBranchParameters
from parameters.level_execution_params import LevelExecutionParameters

class LevelTwoNsight(LevelTwo, LevelOneNsight):
    """
    Class with level two of the execution based on NSIGHT scan tool
    
    Atributes:
        _back_memory_bound      : BackMemoryBoundNsight     ; backs'  memory bound part
        _back_core_bound        : BackCoreBoundNsight       ; backs' core bound part
        _front_band_width       : FrontBandWidthNsight      ; front's bandwith part
        _front_dependency       : FrontDependencyNsight     ; front's dependency part
        _branch_divergence      : DivergenceBranchNsight    ; divergence's branch part
        _replay_divergence      : DivergenceReplayNsight    ; divergence's replay part
    """

    def __init__(self, program : str, input_file : str, output_file : str, output_scan_file : str, collect_metrics : bool,
          front_end : FrontEndNsight, back_end : BackEndNsight, divergence : DivergenceNsight, retire : RetireNsight,
          extra_measure : ExtraMeasureNsight, front_band_width : FrontBandWidthNsight, front_dependency : FrontDependencyNsight,
          back_core_bound : BackCoreBoundNsight, back_memory_bound : BackMemoryBoundNsight):
       
        self._back_core_bound : BackCoreBoundNsight = back_core_bound
        self._back_memory_bound : BackMemoryBoundNsight = back_memory_bound
        self._front_band_width : FrontBandWidthNsight = front_band_width
        self._front_dependency : FrontDependencyNsight = front_dependency 
        self._branch_divergence : DivergenceBranchNsight = DivergenceBranchNsight(DivergenceBranchParameters.C_DIVERGENCE_BRANCH_NAME, DivergenceBranchParameters.C_DIVERGENCE_BRANCH_DESCRIPTION, "")
        self._replay_divergence : DivergenceReplayNsight = DivergenceReplayNsight(DivergenceReplayParameters.C_DIVERGENCE_REPLAY_NAME, DivergenceReplayParameters.C_DIVERGENCE_REPLAY_DESCRIPTION, "")
        super().__init__(program, input_file, output_file, output_scan_file, collect_metrics, front_end, back_end, divergence, retire, extra_measure)
        pass

    def divergence_replay(self) -> DivergenceReplayNsight:
        """
        Return Replay part of the execution.
 
        Returns:
            reference to CoreBound part of the execution
        """
 
        return self._replay_divergence
        pass
 
 
    def divergence_branch(self) -> DivergenceBranchNsight:
        """
        Return Replay part of the execution.
 
        Returns:
            reference to CoreBound part of the execution
        """
        
        return self._branch_divergence        
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
        Return MemoryBound part of the execution.

        Returns:
            reference to MemoryBoundNsight part of the execution
        """
        
        return self._back_memory_bound
        pass

    def front_band_width(self) -> FrontBandWidthNsight:
        """
        Return FrontBandWidth part of the execution.

        Returns:
            reference to FrontBandWidthNsight part of the execution
        """
        
        return self._front_band_width
        pass

    def front_dependency(self) -> FrontDependencyNsight:
        """
        Return FrontDependency part of the execution.

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
        
        command : str = ("ncu --target-processes all --metrics " + self._front_end.metrics_str() +
            "," + self._back_end.metrics_str() + "," + self._divergence.metrics_str() + "," +
            self._extra_measure.metrics_str() + "," + self._retire.metrics_str() + "," +
            self._front_band_width.metrics_str() + "," + self._front_dependency.metrics_str() +
            "," + self._back_core_bound.metrics_str() + "," + self._back_memory_bound.metrics_str() +
             " " + self._program)       
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
    
    def _branch_divergence_ipc_degradation(self) -> float:
        """
        Find IPC degradation due to Divergence.Branch part

        Returns:
            Float with theDivergence's IPC degradation

        """
        
        return super()._branch_diver_ipc_degradation(LevelExecutionParameters.C_WARP_EXECUTION_EFFICIENCY_METRIC_NAME_NSIGHT)
        pass

    def _replay_divergence_ipc_degradation(self) -> float:
        """
        Find IPC degradation due to Divergence.Replay part

        Returns:
            Float with theDivergence's IPC degradation

        """
        
        return super()._replay_diver_ipc_degradation(LevelExecutionParameters.C_ISSUE_IPC_METRIC_NAME_NSIGHT)
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



