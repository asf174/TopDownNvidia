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

class FrontErrorOperation(Exception):
    """Exception raised in FrontEnd launch Command Operation."""
    
    C_ERROR_MESSAGE     : str = "Error in launching FrontEnd operation."

    def __init__(self):
        """Show error message."""

        super().__init__(self.C_ERROR_MESSAGE)

class BackErrorOperation(Exception):
    """Exception raised in BackEnd launch Command Operation."""
    
    C_ERROR_MESSAGE     : str = "Error in launching BackEnd operation."

    def __init__(self):
        """Show error message."""

        super().__init__(self.C_ERROR_MESSAGE)