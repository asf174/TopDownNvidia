"""
Mistakes launched by TopDown class

@author:    Alvaro Saiz (UC)
@date:      Jan 2021
@version:   1.0
"""

class ProfilingError(Exception):
    """Exception raised when NVIDIA scan tool has failed (results not produced)"""
    
    C_ERROR_MESSAGE     : str = "Error with the NVIDIA scan tool (results not generated)"

    def __init__(self):
        """Show error message."""
        
        super().__init__(self.C_ERROR_MESSAGE)
    pass

class MetricNotAsignedToPart(Exception):
    """Exception raised when a metric has not been assigned to any analysis part"""
    
    C_ERROR_MESSAGE     : str = "Following Metric has not been assigned to any analysis part: "

    def __init__(self, metric_name : str):
        """Show error message."""
        
        super().__init__(self.C_ERROR_MESSAGE + metric_name)
    pass

class EventNotAsignedToPart(Exception):
    """Exception raised when an event has not been assigned to any analysis part"""
    
    C_ERROR_MESSAGE     : str = "Following event has not been assigned to any analysis part: "

    def __init__(self, event_name : str):
        """Show error message."""
        
        super().__init__(self.C_ERROR_MESSAGE + event_name)
    pass

class WriteInOutPutFileError(Exception):
    """Exception raised in case of error writing in output file"""
    
    C_ERROR_MESSAGE     : str = "Error writing in output file"

    def __init__(self):
        """Show error message."""
        
        super().__init__(self.C_ERROR_MESSAGE)
    pass

class MetricNoDefined(Exception):
    """Exception raised when a metric has introduced but does not exist in NVIDIA scan tool.
    
    Attributes:
        metric_name    : str   ; name of the metric that produced the error
    """
    
    C_ERROR_MESSAGE     : str = "Error. Following Metric does not exist in NVIDIA scan tool: "

    def __init__(self, metric_name: str):
        """Show error message."""
        
        super().__init__(self.C_ERROR_MESSAGE + metric_name)
    pass

class EventNoDefined(Exception):
    """Exception raised when a event has introduced but does not exist in NVIDIA scan tool.
    
    Attributes:
        event_name    : str   ; name of the event that produced the error
    """
    
    C_ERROR_MESSAGE     : str = "Error. Following event does not exist in NVIDIA scan tool: "

    def __init__(self, event_name: str):
        """Show error message."""
        
        super().__init__(self.C_ERROR_MESSAGE + event_name)
    pass

