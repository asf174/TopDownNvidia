"""
Mistakes launched by TopDown class

@author:    Alvaro Saiz (UC)
@date:      Jan 2021
@version:   1.0
"""


class WriteInOutPutFileError(Exception):
    """Exception raised in case of error writing in output file"""
    
    C_ERROR_MESSAGE     : str = "Error writing in output file"

    def __init__(self):
        """Show error message."""
        
        super().__init__(self.C_ERROR_MESSAGE)
        pass
    
class ComputeCapabilityError(Exception):
    """Exception raised when compute capability of (current) device cannot be obtained
    
    Attributes:
        event_name    : str   ; name of the event that produced the error
    """
    
    C_ERROR_MESSAGE     : str = "Error obtaining Compute Capability of current device"

    def __init__(self):
        """Show error message."""
        
        super().__init__(self.C_ERROR_MESSAGE)
        pass

