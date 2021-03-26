from abc import ABC, abstractmethod # abstract class
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from measure_levels.level_one import LevelOne 

class LevelOneNsight(LevelOne, LevelExecutionNsight):


    """ 
    Class thath represents the level one of the execution with nsight scan tool.

    Attributes:
        _front_end      : FrontEnd      ; FrontEnd part of the execution
        _back_end       : BackEnd       ; BackEnd part of the execution
        _divergence     : Divergence    ; Divergence part of the execution
        _retire         : Retire        ; Retire part of the execution
    """

    def __init__(self, program : str, output_file : str, recoltect_metrics : bool):
        self._front_end : FrontEndNsight = FrontEndNsight()
        self._back_end  : BackEndNsight = BackEndNsight()
        self._divergence : DivergenceNsight = DivergenceNsight()
        self._retire : RetireNsight = RetireNsight()
        super(LevelExecutionNvprof, self).__init__(program, output_file, recoltect_metrics)
        pass

    def _generate_command(self) -> str:
        """ 
        Generate command of execution with NVIDIA scan tool.

        Returns:
            String with command to be executed
        """
        
        command : str = ("sudo $(which ncu) --page=raw --metrics " + self._front_end.metrics_str() + 
            "," + self._back_end.metrics_str() + "," + self._divergence.metrics_str() + "," + self._extra_measure.metrics_str()
            + "," + self._retire.metrics_str() + self._program)
        return command
        pass

    def front_end(self) -> FrontEnd:
        """
        Return FrontEndNsight part of the execution.

        Returns:
            reference to FrontEndNsight part of the execution
        """

        return self._front_end
        pass
    
    def back_end(self) -> BackEnd:
        """
        Return BackEndNsight part of the execution.

        Returns:
            reference to BackEndNsight part of the execution
        """

        return self._back_end
        pass

    def divergence(self) -> Divergence:
        """
        Return DivergenceNsight part of the execution.

        Returns:
            reference to DivergenceNsight part of the execution
        """

        return self._divergence
        pass

    def retire(self) -> Retire:
        """
        Return RetireNsight part of the execution.

        Returns:
            reference to RetireNsight part of the execution
        """

        return self._retire
        pass

    def __divergence_ipc_degradation(self) -> float:
        """
        Find IPC degradation due to Divergence part

        Returns:
            Float with theDivergence's IPC degradation

        """
        return super()._diver_ipc_degradation(LevelExecutionParameters.C_WARP_EXECUTION_EFFICIENCY_NAME_NSIGHT)
        pass

    def _get_results(self, lst_output : list[str]):
        """
        Get results of the different parts.

        Parameters:
            lst_output              : list[str]     ; OUTPUT list with results
        """

        converter : MessageFormat = MessageFormat()
        #  Keep Results
        if not self._recolect_metrics:
            return
        if self._recolect_metrics and self._front_end.metrics_str() != "":
            lst_output.append(converter.underlined_str(self._front_end.name()))
            super()._add_result_part_to_lst(self._front_end.metrics(), 
                self._front_end.metrics_description(), lst_output, True)           
        if self._recolect_metrics and self._back_end.metrics_str() != "":
            lst_output.append(converter.underlined_str(self._back_end.name()))
            super()._add_result_part_to_lst(self._back_end.metrics(), 
                self._back_end.metrics_description(), lst_output, True)  
        if self._recolect_metrics and self._divergence.metrics_str() != "":
            lst_output.append(converter.underlined_str(self._back_end.name()))
            super()._add_result_part_to_lst(self._divergence.metrics(), 
                self._divergence.metrics_description(), lst_output, True)
        if self._recolect_metrics and self._retire.metrics_str() != "":
            lst_output.append(converter.underlined_str(self._retire.name()))
            super()._add_result_part_to_lst(self._retire.metrics(), 
                self._retire.metrics_description(), lst_output, True)
        if self._recolect_metrics and self._extra_measure.metrics_str() != "":
            lst_output.append(converter.underlined_str(self._extra_measure.name()))
            super()._add_result_part_to_lst(self._extra_measure.metrics(), 
                self._extra_measure.metrics_description(), lst_output, True)
        lst_output.append("\n")
        pass
