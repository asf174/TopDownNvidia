"""
Measurements made by the TopDown methodology.

@author:    Alvaro Saiz (UC)
@date:      Jan-2021
@version:   1.0
"""

import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from errors.metric_measure_errors import * 
from abc import ABC # abstract class

class MetricMeasure(ABC):
    """
    Class that implements the metrics used 
    to analyze the TopDown methodology over NVIDIA's GPUs.
    
    Attributes:
        _name              : str   ;   measure name.
        
        _description       : str   ;   description with information.
        
        _metrics           : dict  ;   dictionary with metric name as key, 
                                        and value of metric as value.
        
        _events            : dict  ;   dictionary with metric name as key, 
                                        and value of event as value.
        
        _metrics_desc      : dict  ;   dictionary with metric name as key, 
                                        and description of metric as value.
        
        _events_desc       : dict  ;   dictionary with event name as key, 
                                        and description of event as value.

        _metrics_str       : str   ;   string with the metrics
        _events_str        : str   ;   string with the events
    """

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
                
    def __init__(self, name : str, description : str, metrics : str, events : str, metrics_desc : str, events_desc : str):
        """
        Set attributtes with argument values.
        
        Params:
            
            name                : str   ;   measure name.
        
            description         : str   ;   description with information.
        
            metrics             : str   ;   string with the metrics
        
            events              : str   ;   string with events
        
            metrics_desc        : str   ;   string with metric name as key, 
                                            and description of metric as value.
        
            events_desc         : str   ;   dictionary with event name as key, 
                                            and description of event as value.
        """

        self._name : str = name
        self._description : str = description

        self._metrics : dict = dict()
        self._metrics_desc : dict = dict()
        if metrics != "":
            self._metrics  = dict.fromkeys(metrics.split(","))
            self._metrics_desc = dict.fromkeys(metrics_desc.split(","))
        
        self._events : dict = dict()
        self._events_desc : dict = dict()
        if events != "":
            self._events = dict.fromkeys(events.split(","))
            self._events_desc = dict.fromkeys(events_desc.split(","))
        
        self._metrics_str : str = metrics
        self._events_str : str = events
        self._check_data_structures() # check dictionaries defined correctly
        self.__init_dictionaries()
        pass
    
    def _check_data_structures(self): 
        """
        Check all data structures for events and metrics are consistent.
        In case they are not, raise exception.

        Raises:
            DataStructuresOfEventError  ; if event is not defined in all data structures
            DataStructuresOfMetricError ; if metric is not defined in all data structures
        """

        event_name : str
        metric_name : str
        for (event_name, value), (metric_name, value) in zip(self._events.items(), self._metrics.items()): 
            if not event_name in self._events_desc:
                raise DataStructuresOfEventError(event_name)
            if not metric_name in self._metrics_desc:
                raise DataStructuresOfMetricError(metric_name)
        pass

    def is_event(self, event_name : str) -> bool:
        """
        Check if argument it's an event or not

        Params:
            event_name  : str   ; name of the event

        Returns:
            True if 'event_name' it's a correct event or False in other case
        """

        is_in_events : bool = event_name in self._events
        is_in_events_desc : bool = event_name in self._events_desc
        
        if not (is_in_events and is_in_events_desc):
            return False
        return True
        pass

    def is_metric(self, metric_name : str) -> bool:
        """
        Check if argument it's a metric or not

        Params:
            event_name  : str   ; name of the metric

        Returns:
            True if 'event_name' it's a correct event or False in other case
        """

        is_in_metrics : bool = metric_name in self._metrics
        is_in_metrics_desc : bool = metric_name in self._metrics_desc
        
        if not (is_in_metrics and is_in_metrics_desc):
            return False
        return True
        pass
    
    def get_event_value(self, event_name : str) -> list[str]:
        """
        Get the value/s associated with 'event_name'

        Params:
            event_name  : str   ; name of the event

        Returns:
            List with associated value/s to 'event_name' or 'None' if
            'event_name' doesn't exist or it's not an event

        """

        if self.is_metric(event_name) or not self.is_event(event_name):
            return None
        return self._events.get(event_name)
        pass

    def get_event_description(self, event_name : str) -> list[str]:
        """
        Get the description/s associated with 'event_name'

        Params:
            event_name  : str   ; name of the event

        Returns:
            List with associated description/s to 'event_name' or 'None' if
            'event_name' doesn't exist or it's not an event

        """

        if self.is_metric() or not self.is_event():
            return None
        return self._events_desc.get(event_name)
        pass

    def get_metric_value(self, metric_name : str) -> list[str]:
        """
        Get the value/s associated with 'metric_name'

        Params:
            metric_name  : str   ; name of the metric

        Returns:
            List with associated value/s 'metric_name' or 'None' if
            'metric_name' doesn't exist or it's not a metric

        """

        if not self.is_metric(metric_name) or self.is_event(metric_name):
            return None
        return self._metrics.get(metric_name)
        pass

    def get_metric_description(self, metric_name : str) -> list[str]:
        """
        Get the description/s associated with 'metric_name'

        Params:
            metric_name  : str   ; name of the metric

        Returns:
            List with associated description/s to 'metric_name' or 'None' if
            'metric_name' doesn't exist or it's not a metric

        """
        
        if not self.is_metric(metric_name) or self.is_event(metric_name):
            return None
        return self._metrics_desc.get(metric_name)
        pass

    def set_metric_value(self, metric_name : str, new_value : str) -> bool:
        """
        Update metric with key 'metric_name' with 'new_value' value if 'metric_name' exists.

        Params:
            metric_name     : str   ; name of the metric
            new_value       : str   ; new value to assign to 'metric_name' if name exists
        
        Returns:
            True if the operation was perfomed succesfully or False if not because 'metric_name'
            does not correspond to any metric
        """

        if not (metric_name in self._metrics):
            return False
        
        self._metrics[metric_name].append(new_value)
        return True
        pass

    def set_metric_description(self, metric_name : str, new_description : str) -> bool:
        """
        Update metric with key 'metric_name' with 'new_value' description if 'metric_name' exists.
        Params:
            metric_name         : str   ; name of the metric
            new_description     : str   ; new description to assign to 'metric_name' if name exists
        
        Returns:
            True if the operation was perfomed succesfully or False if not because 'metric_name'
            does not correspond to any metric
        """

        if not metric_name in self._metrics:
            return False
        self._metrics_desc[metric_name] = new_description
        return True
        pass

    def set_event_value(self, event_name : str, new_value : str) -> bool:
        """
        Update event with key 'event_name' with 'new_value' value if 'event_name' exists.

        Params:
            event_name     : str   ; name of the event
            new_value       : str   ; new value to assign to 'event_name' if name exists
        
        Returns:
            True if the operation was perfomed succesfully or False if not because 'event_name'
            does not correspond to any event
        """

        if not (event_name in self._events):
            return False
        self._events[event_name].append(new_value)
        return True
        pass

    def set_event_description(self, event_name : str, new_description : str) -> bool:
        """
        Update event with key 'event_name' with 'new_value' description if 'event_name' exists.
        Params:
            event_name     : str   ; name of the event
            new_value       : str   ; new description to assign to 'event_name' if name exists
        
        Returns:
            True if the operation was perfomed succesfully or False if not because 'event_name'
            does not correspond to any event
        """

        if not (event_name in self._event):
            return False
        self._events_desc[event_name] = new_description
        return True
        pass

    def name(self) -> str:
        """ 
        Return measure name.
        
        Returns:
            String with the measure name
        """

        return self._name
        pass
    
    def description(self) -> str:
        """ 
        Return the description with information.
        
        Returns:
            String with the description
        """
        
        return self._description
        pass

    def metrics(self) -> dict: 
        """ 
        Return the metrics and their values.
        
        Returns:
            Dictionary with the metrics and their values
        """

        return self._metrics # mirar retornar copias
        pass

    def metrics_description(self) -> dict: 
        """ 
        Return the metrics and their descriptions.
        
        Returns:
            Dictionary with the metrics and their descriptions
        """

        return self._metrics_desc # mirar retornar copias
        pass

    def events(self) -> dict: 
        """ 
        Return the events and their values.
        
        Returns:
            Dictionary with the events and their values
        """
        
        return self._events # mirar retornar copias
        pass

    def events_description(self) -> dict: 
        """ 
        Return the events and their descriptions.
        
        Returns:
            Dictionary with the events and their descriptions
        """

        return self._events_desc # mirar retornar copias
        pass

    def metrics_str(self) -> str:
        """ 
        Returns a string with the metrics

        Returns:
            String with the metrics
        """

        return self._metrics_str
        pass

    def events_str(self) -> str:
        """ 
        Returns a string with the events

        Returns:
            String with the events
        """

        return self._events_str
        pass