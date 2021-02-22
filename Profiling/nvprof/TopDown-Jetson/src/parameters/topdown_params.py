"""
Class with all params of topdown.py file

@author:    Alvaro Saiz (UC)
@date:      Jan 2021
@version:   1.0
"""

class TopDownParameters:
  
    """
    Options of program
    """
    # Level
    C_LEVEL_SHORT_OPTION                : str       = "-l"
    C_LEVEL_LONG_OPTION                 : str       = "--level"
    
    # Long description
    C_LONG_DESCRIPTION_SHORT_OPTION     : str       = "-ld"
    C_LONG_DESCRIPTION_LONG_OPTION      : str       = "--long-desc"
    
    # Help 
    C_HELP_SHORT_OPTION                 : str       = "-h"
    C_HELP_LONG_OPTION                  : str       = "--help"

    # Output file
    C_OUTPUT_FILE_SHORT_OPTION          : str       = "-o"
    C_OUTPUT_FILE_LONG_OPTION           : str       = "--output"

    # Program file
    C_FILE_SHORT_OPTION                 : str       = "-f"
    C_FILE_LONG_OPTION                  : str       = "--file"

    """Options."""
    C_MIN_LEVEL_EXECUTION               : int       = 1
    C_MAX_LEVEL_EXECUTION               : int       = 2



