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
    C_VERBOSE_SHORT_OPTION              : str       = "-v"
    C_VERBOSE_LONG_OPTION               : str       = "--verbose"
    
    # Help 
    C_HELP_SHORT_OPTION                 : str       = "-h"
    C_HELP_LONG_OPTION                  : str       = "--help"

    # Output file
    C_OUTPUT_FILE_SHORT_OPTION          : str       = "-o"
    C_OUTPUT_FILE_LONG_OPTION           : str       = "--output"

    # Program file
    C_FILE_SHORT_OPTION                 : str       = "-f"
    C_FILE_LONG_OPTION                  : str       = "--file"

    # delete content
    C_DELETE_CONTENT_SHORT_OPTION       : str       = "-dc"
    C_DELETE_CONTENT_LONG_OPTION        : str       = "--delete-content"

    # show description
    C_SHOW_DESCRIPTION_SHORT_OPTION     : str       = "-nd"
    C_SHOW_DESCRIPTION_LONG_OPTION      : str       = "--no-desc"

    C_MIN_LEVEL_EXECUTION               : int       = 1
    C_MAX_LEVEL_EXECUTION               : int       = 3

    # recolect metrics
    C_METRICS_SHORT_OPTION              : str       = "-m"
    C_METRICS_LONG_OPTION               : str       = "--metrics"

    # recolect events
    C_EVENTS_SHORT_OPTION               : str       = "-e"
    C_EVENTS_LONG_OPTION                : str       = "--events"

    # recolect all measures
    C_ALL_MEASURES_SHORT_OPTION         : str       = "-am"
    C_ALL_MEASURES_LONG_OPTION          : str       = "--all-measurements" # NOMBRE: C_ALL_MEASURES_ARGUMENT_LONG_OPTION TODO

    C_INTRO_MESSAGE_GENERAL_INFORMATION : str       = ("\n- Program Name:    topdown.py\n" + \
                                                        "- Author:          Alvaro Saiz (UC)\n" + \
                                                        "- Contact info:    asf174@alumnos.unican.es\n" + \
                                                        "- Company:         University Of Cantabria\n" + \
                                                        "- Place:           Santander, Cantabria, Kingdom of Spain\n" + \
                                                        "- Advisors:        Pablo Abad (UC) <pablo.abad@unican.es>, Pablo Prieto (UC) <pablo.prieto@unican.es>\n" + \
                                                        "- Bugs Report:     asf174@alumnos.unican.es"+ 
                                                        "\n\n- Licence:         GNU GPL")

    C_NUM_MAX_CHARACTERS_PER_LINE       : int       = 129

    C_MAX_NUM_RESULTS_DECIMALS          : int       = 3

    # arch device
    C_TESLA_DEVICE_SHORT_OPTION         : str       = "-t"
    C_TESLA_DEVICE_LONG_OPTION          : str       = "--tesla"  
