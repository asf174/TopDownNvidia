from abc import ABC, abstractmethod # abstract class
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from measure_parts.extra_measure import ExtraMeasure    
from shell.shell import Shell # launch shell arguments
from parameters.level_execution_params import LevelExecutionParameters # parameters of program
from errors.level_execution_errors import *
from measure_levels.level_execution import LevelExecution 
from measure_parts.extra_measure import ExtraMeasureNvprof

class LevelExecutionNvprof(LevelExecution, ABC):
    """ 
    Class that represents the levels of the execution with nvprof scan tool
     
    Attributes:
         _extra_measure      : ExtraMeasureNvprof  ; support measures
        _recolect_events    : bool          ; True if the execution must recolted the events used by NVIDIA scan tool
                                              or False in other case
    """

    def __init__(self, program : str, output_file : str, recoltect_metrics : bool, recolect_events : bool):
        self._extra_measure : ExtraMeasureNvprof = ExtraMeasureNvprof()
        self._recolect_events = recolect_events
        super().__init__(program, output_file, recoltect_metrics)
        pass

    def recolect_events(self) -> bool:
        """
        Check if execution must recolect NVIDIA's scan tool events.

        Returns:
            Boolean with True if it has to recolect events or False if not
        """

        return self._recolect_events
        pass

    def extra_measure(self) -> ExtraMeasureNvprof:
        """
        Return ExtraMeasureNvprof part of the execution.

        Returns:
            reference to ExtraMeasureNvprof part of the execution
        """
        
        return self._extra_measure
        pass

    @abstractmethod
    def run(self, lst_output : list):
        """
        Makes execution.
        
        Parameters:
            lst_output  : list ; list with results
        """
        
        pass
  
    @abstractmethod
    def _generate_command(self) -> str:
        """ 
        Generate command of execution with NVIDIA scan tool.

        Returns:
            String with command to be executed
        """

        pass

    @abstractmethod
    def _get_results(self, lst_output : list):
        """ 
        Get results of the different parts.

        Parameters:
            lst_output              : list     ; OUTPUT list with results
        """

        pass

    def _add_result_part_to_lst(self, dict_values : dict, dict_desc : dict,
        lst_to_add , isMetric : bool):
        """
        Add results of execution part (FrontEnd, BackEnd...) to list indicated by argument.

        Params:
            dict_values     : dict      ; diccionary with name_metric/event-value elements of the part to
                                          add to 'lst_to_add'
            dict_desc       : dict      ; diccionary with name_metric/event-description elements of the
                                          part to add to 'lst_to_add'
            lst_output      : list ; list where to add all elements
            isMetric        : bool      ; True if they are metrics or False if they are events

        Raises:
            MetricNoDefined             ; raised in case you have added an metric that is
                                          not supported or does not exist in the NVIDIA analysis tool
            EventNoDefined              ; raised in case you have added an event that is
                                          not supported or does not exist in the NVIDIA analysis tool
        """

        # metrics_events_not_average : list = LevelExecutionParameters.C_METRICS_AND_EVENTS_NOT_AVERAGE_COMPUTED.split(",")
        metrics_events_not_average  = LevelExecutionParameters.C_METRICS_AND_EVENTS_NOT_AVERAGE_COMPUTED.split(",")
        total_value : float = 0.0
        description : str
        line_lenght : int
        value_str : str
        total_value_str : str = ""
        value_measure_str : str

        measure_name : str = "Event"
        desc_name_title : str = measure_name + " Name"
        desc_max_length : int = len(desc_name_title)
        if isMetric:
            measure_name = "Metric"
            desc_name_title = measure_name + " Description"
            desc_max_length = len(desc_name_title)
            for key_desc in dict_desc:
                if len(key_value[0]) > desc_max_length: #TODO este dict solo tiene que tener un valor
                    desc_max_length = key_value[0]
        
        measure_name_title : str = measure_name + " Name"
        desc_max_length : int = len(metric_desc_title)
        measure_value_title : str = measure_name + " Value"
        
        for key_value in dict_values:
            if len(key_value) > name_max_length:
                name_max_length = len(key_value)
                name_max_length : int = len(measure_name + " Name")
        
        description = ("\t\t\t%-*s" % (metric_name_length , "Metric Name"))
        description += ("%-*s" % (metric_unit_length , metric_unit_title))
        description += ("%-*s" % (metric_value_length, metric_value_title))

        if isMetric:
            metric_name : str
            value : float
            is_percentage : bool = False
            is_computed_as_average : bool
            for key_value,key_desc in zip(dict_values, dict_desc):
                if dict_values[key_value][0][len(dict_values[key_value][0]) - 1] == "%":
                    is_percentage = True
                    # In NVIDIA scan tool, the percentages in each kernel are calculated on the total of
                    # each kernel and not on the total of the application
                    if key_value in metrics_events_not_average:
                        raise ComputedAsAverageError(key_value)
                is_computed_as_average = not (key_value in metrics_events_not_average)
                total_value = round(self._get_total_value_of_list(dict_values[key_value], is_computed_as_average),
                    LevelExecutionParameters.C_MAX_NUM_RESULTS_DECIMALS)
                if total_value.is_integer():
                    total_value = int(total_value)
                value_measure_str = str(total_value)
                if is_percentage:
                    value_measure_str += "%"
                    is_percentage = False
                metric_name = key_value
                metric_desc = dict_desc.get(key_desc)
                value_str = ("\t\t\t%-*s" % (len(metric_name) + metric_name_length - len(metric_name) , metric_name))
                value_str += ("%-*s" % (len(metric_desc) + metric_desc_length - len(metric_desc) , metric_desc))
                value_str += ("%-*s" % (len(metric_value_title), value_measure_str))
                if len(value_str) > line_lenght:
                    line_lenght = len(value_str)
                total_value_str += value_str
        else:
            event_name : str
            for key_value in dict_values:
                total_value = round(self._get_total_value_of_list(dict_values[key_value], False),
                    LevelExecutionParameters.C_MAX_NUM_RESULTS_DECIMALS) # TODO eventos con decimales?
                if total_value.is_integer():
                    total_value = int(total_value)
                value_measure_str = str(total_value)
                event_name = key_value
                value_str = ("\t\t\t%-*s" % (len(event_name) + metric_name_length - len(event_name) , event_name))
                value_str += ("%-*s" % (len("-") + metric_desc_length - len(metric_desc) , "-"))
                value_str += ("%-*s" % (len(metric_value_title), value_measure_str))
                if len(value_str) > line_lenght:
                    line_lenght = len(value_str)
                total_value_str += value_str
            
        total_value_str += "\n" 
        spaces_lenght : int = len("\t\t\t")
        line_str : str = "\t\t\t" + f'{"-" * (line_lenght - spaces_lenght)}'
        lst_to_add.append("\n" + line_str)
        lst_to_add.append(description)
        lst_to_add.append(line_str)
        lst_to_add.append(total_value_str)
        lst_to_add.append(line_str + "\n")
        pass    

    """  metric_name_length : int = name_max_length + 10
        metric_unit_length : int = unit_max_length + 10
        metric_value_length : int = len(metric_value_title)
        
        description = ("\t\t\t%-*s" % (metric_name_length , "Metric Name"))
        description += ("%-*s" % (metric_unit_length , metric_unit_title))
        description += ("%-*s" % (metric_value_length, metric_value_title))
                   

        
        line_lenght = len(description) 
        for key_value, key_unit in zip(dict_values, dict_desc):
            total_value = round(self._get_total_value_of_list(dict_values[key_value], False),
             LevelExecutionParameters.C_MAX_NUM_RESULTS_DECIMALS) # TODO eventos con decimales? 
            if total_value.is_integer():
                total_value = int(total_value)
            value_metric_str = str(total_value)
            metric_name = key_value
            metric_unit = dict_desc.get(key_unit)
            if not metric_unit:
                metric_unit = "-"
            elif metric_unit == "%":
                # In NVIDIA scan tool, the percentages in each kernel are calculated on the total of
                # each kernel and not on the total of the application
                if key_value in metrics_events_not_average:
                    raise ComputedAsAverageError(key_unit)

            value_str = ("\t\t\t%-*s" % (len(metric_name) + metric_name_length - len(metric_name) , metric_name))
            value_str += ("%-*s" % (len(metric_unit) + metric_unit_length - len(metric_unit) , metric_unit))
            value_str += ("%-*s" % (len(metric_value_title), value_metric_str))
        
            if len(value_str) > line_lenght:
                line_lenght = len(value_str)
            if i != len(dict_values) -1:
                value_str += "\n"
            total_value_str += value_str
            i += 1
        spaces_lenght : int = len("\t\t\t")
        line_str : str = "\t\t\t" + f'{"-" * ((line_lenght - spaces_lenght))}'
        lst_to_add.append("\n" + line_str)
        lst_to_add.append(description)
        lst_to_add.append(line_str)
        lst_to_add.append(total_value_str)
        lst_to_add.append(line_str + "\n")
    """
    def _percentage_time_kernel(self, kernel_number : int) -> float:
        """ 
        Get time percentage in each Kernel.
        Each kernel measured is an index of dictionaries used by this program.

        Params:
            kernel_number   : int   ; number of kernel
        """
        
        value_lst : list = self._extra_measure.get_event_value(LevelExecutionParameters.C_CYCLES_ELAPSED_EVENT_NAME_NVPROF)
        if value_lst is None:
            raise ElapsedCyclesError
        value_str : str
        total_value : float = 0.0
        for value_str in value_lst:
            total_value += float(value_str)
        return (float(value_lst[kernel_number])/total_value)*100.0
        pass

