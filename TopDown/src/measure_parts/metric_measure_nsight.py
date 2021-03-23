"""
Measurements made by the TopDown methodology with nsight scan tool.

@author:    Alvaro Saiz (UC)
@date:      Jan-2021
@version:   1.0
"""

import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from errors.metric_measure_errors import * 

class MetricMeasureNsight(MetricMeasure):
    """
    Class that implements the metrics used 
    to analyze the TopDown methodology over NVIDIA's GPUs with
    with nvprof scan tool.
    """

    @override
    def __init_dictionaries(self):
        """ Initialize data structures in the correct way."""

        key_metrics : str
        key_events : str
        key_metrics_desc : str
        key_events_desc : str
        # TODO todo esto en un bucle for con 'zip'
        for key_metrics in self._metrics:
            self._metrics[key_metrics] = list()
        for key_events in self._events:
            self._events[key_events] = list()
        pass
                
    def __init__(self, name : str, description : str, metrics : str, metrics_desc : str):
        """
        Set attributtes with argument values.
        
        Params:
            
            name                : str   ;   measure name.
        
            description         : str   ;   description with information.
        
            metrics             : str   ;   string with the metrics
         
            metrics_desc        : str   ;   string with metric name as key, 
                                            and description of metric as value.
        """
        
        super().__init__(name, description, metrics, metrics_desc)    
        pass