"""
Errores launched by TopDown class

@author:    Alvaro Saiz (UC)
@date:      Jan 2021
@version:   1.0
"""

class FrontAndBackErrorOperation(Exception):
    """Exception raised in FrontEnd and BackEnd launch Command Operation."""
    
    C_ERROR_MESSAGE     : str = "Error in launching FrontEnd and BackEnd operations."

    def __init__(self):
        """Show error message."""
        
        super().__init__(self.C_ERROR_MESSAGE)
    pass

class OutputResultCommandError(Exception):
    """Exception raised in case of error reading the output produced by command."""
    
    C_ERROR_MESSAGE     : str = "Error reading the output produced by command"

    def __init__(self):
        """Show error message."""
        
        super().__init__(self.C_ERROR_MESSAGE)
    pass

class WriteInOutPutFileError(Exception):
    """Exception raised in case of error writing in output file"""
    
    C_ERROR_MESSAGE     : str = "Error writing in output file"

    def __init__(self):
        """Show error message."""
        
        super().__init__(self.C_ERROR_MESSAGE)
    pass

class MetricOrEventNoDefined(Exception):
    """Exception raised when a metric/event has introduced but does not exist in NVIDIA scan tool.
    
    Attributes:
        name_counter    : str   ; name of the counter that produced the error
    """
    
    C_ERROR_MESSAGE     : str = "Error. Following Metric/event does not exist in NVIDIA scan tool: "

    def __init__(self, name_counter: str):
        """Show error message."""
        
        super().__init__(self.C_ERROR_MESSAGE + name_counter)
    pass
