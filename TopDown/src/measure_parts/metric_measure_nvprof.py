"""
Measurements made by the TopDown methodology with nvprof scan tool.

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

class MetricMeasureNvprof(MetricMeasure):
    """
    Class that implements the metrics and events used 
    to analyze the TopDown methodology over NVIDIA's GPUs with
    with nvprof scan tool.

    Attributes:
        __events            : dict  ;   dictionary with event name as key, 
                                        and value of event as value.
        
        __events_desc       : dict  ;   dictionary with events name as key, 
                                        and description of events as value.
                                        
        __events_str        : str   ;   string with the events
    """

    @override
    def __init_dictionaries(self, events : str, events_desc : str):
        """ 
        Initialize data structures in the correct way.
        
        Params:

            events              : str   ;   string with events

            events_desc         : str   ;   dictionary with event name as key, 
                                            and description of event as value.
        """

        if events != "":
            self.__events = dict.fromkeys(events.replace(" ", "").split(","))
            self.__events_desc = dict.fromkeys(events_desc.replace(" ", "").split(","))
        
        self.__events_str : str = events

        key_events : str
        key_events_desc : str
        for key_events in self._events:
            self._events[key_events] = list()
        pass

    @override
    def __check_data_structures(self): 
        """
        Check all data structures for events and metrics are consistent.
        In case they are not, raise exception.

        Raises:
            DataStructuresOfEventError  ; if event is not defined in all data structures
        """

        event_name : str
        for event_name in self.__events.items(): 
            if not metric_name in self.__events_desc:
                raise DataStructuresOfEventError(metric_name)
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
        super().__init__(name, description, metrics, metrics_desc)

        self.__init_dictionaries()
        self.__check_data_structures() # check dictionaries defined correctly
        pass
    
    def __init__(self, metrics : str, metrics_desc : str):
        """
        Set attributtes with argument values.
        
        Params:
            
            name                : str   ;   measure name.
        
            description         : str   ;   description with information.
        
            metrics             : str   ;   string with the metrics
            
            metrics_desc        : str   ;   string with metric name as key, 
                                            and description of metric as value. 
        """

        super().__init__(metrics, metrics_desc)
        self.__init_dictionaries()
        self.__check_data_structures() # check dictionaries defined correctly
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

    def events_str(self) -> str:
        """ 
        Returns a string with the events

        Returns:
            String with the events
        """

        return self._events_str
        pass