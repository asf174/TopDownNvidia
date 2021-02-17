"""
Measurements made by the TopDown methodology in back end part.

@author:    Alvaro Saiz (UC)
@date:      Jan-2021
@version:   1.0
"""
import sys
sys.path.insert(1, '../errors/')
sys.path.insert(1, '../parameters/')

from metric_base_errors import * 
from metric_base_params import MetricBaseParameters # TODO IMPORT ONLY ATTRIBUTES
from metric_base import MetricBase


class BackEnd(MetricBase):
    """Class that defines the Back-End part."""

    def __init__(self):
        """Set attributes with DEFAULT values."""

        self._name : str = MetricBaseParameters.C_BACK_END_NAME
        self._description : str = MetricBaseParameters.C_BACK_END_DESCRIPTION

        if MetricBaseParameters.C_BACK_END_METRICS != "": # empty
            self._metrics : dict = dict.fromkeys(MetricBaseParameters.C_BACK_END_METRICS.split(","))
            self._metrics_desc : dict = dict.fromkeys(MetricBaseParameters.C_BACK_END_METRICS.split(","))
        else:
            self._metrics : dict = dict()
            self._metrics_desc : dict = dict()

        if MetricBaseParameters.C_BACK_END_EVENTS != "":
            self._events : dict = dict.fromkeys(MetricBaseParameters.C_BACK_END_EVENTS.split(","))
            self._events_desc : dict = dict.fromkeys(MetricBaseParameters.C_BACK_END_EVENTS.split(","))
        else:
            self._events : dict = dict()
            self._events_desc : dict = dict()

        self._metrics_str : str = MetricBaseParameters.C_BACK_END_METRICS
        self._events_str : str = MetricBaseParameters.C_BACK_END_EVENTS
        
        super()._check_data_structures() # check dictionaries defined correctly
    pass
pass