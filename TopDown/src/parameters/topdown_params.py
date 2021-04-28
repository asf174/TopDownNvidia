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
    C_LEVEL_ARGUMENT_SHORT_OPTION                           : str       = "-l"
    C_LEVEL_ARGUMENT_LONG_OPTION                            : str       = "--level"
    C_LEVEL_ARGUMENT_DESCRIPTION                            : str       = "level of execution"

    # Long description
    C_VERBOSE_ARGUMENT_SHORT_OPTION                         : str       = "-v"
    C_VERBOSE_ARGUMENT_LONG_OPTION                          : str       = "--verbose"
    C_VERBOSE_ARGUMENT_DESCRIPTION                          : str       = "long description of results"

    # Help 
    C_HELP_ARGUMENT_SHORT_OPTION                            : str       = "-h"
    C_HELP_ARGUMENT_LONG_OPTION                             : str       = "--help"

    # Output file
    C_OUTPUT_FILE_ARGUMENT_SHORT_OPTION                     : str       = "-o"
    C_OUTPUT_FILE_ARGUMENT_LONG_OPTION                      : str       = "--output"
    C_OUTPUT_FILE_ARGUMENT_DESCRIPTION                      : str       = "output file. Path to file."
    
    # Output file
    C_OUTPUT_GRAPH_FILE_ARGUMENT_SHORT_OPTION               : str       = "-og"
    C_OUTPUT_GRAPH_FILE_ARGUMENT_LONG_OPTION                : str       = "--output-graph"
    C_OUTPUT_GRAPH_FILE_ARGUMENT_DESCRIPTION                : str       = "output graph file. Path to file."


    # Program file
    C_FILE_ARGUMENT_SHORT_OPTION                            : str       = "-f"
    C_FILE_ARGUMENT_LONG_OPTION                             : str       = "--file"
    C_FILE_ARGUMENT_DESCRIPTION                             : str       = "run file. Path to file."

    # delete content
    C_DELETE_CONTENT_ARGUMENT_SHORT_OPTION                  : str       = "-dc"
    C_DELETE_CONTENT_ARGUMENT_LONG_OPTION                   : str       = "--delete-content"
    C_DELETE_CONTENT_ARGUMENT_DESCRIPTION                   : str       = "If '-o/--output' is set delete output's file contents before write results"
    # show description
    C_SHOW_DESCRIPTION_ARGUMENT_SHORT_OPTION                : str       = "-nd"
    C_SHOW_DESCRIPTION_ARGUMENT_LONG_OPTION                 : str       = "--no-desc"
    C_SHOW_DESCRIPTION_ARGUMENT_DESCRIPTION                 : str       = "don't show description of results." 

    C_MIN_LEVEL_EXECUTION                                   : int       = 1
    C_MAX_LEVEL_EXECUTION                                   : int       = 3

    # recolect metrics
    C_METRICS_ARGUMENT_SHORT_OPTION                         : str       = "-m"
    C_METRICS_ARGUMENT_LONG_OPTION                          : str       = "--metrics"
    C_METRICS_ARGUMENT_DESCRIPTION                          : str       = "show metrics computed by NVIDIA scan tool" 

    # recolect events
    C_EVENTS_ARGUMENT_SHORT_OPTION                          : str       = "-e"
    C_EVENTS_ARGUMENT_LONG_OPTION                           : str       = "--events"
    C_EVENTS_ARGUMENT_DESCRIPTION                           : str       = "show eventss computed by NVIDIA scan tool"

    # recolect all measures
    C_ALL_MEASURES_ARGUMENT_SHORT_OPTION                    : str       = "-am"
    C_ALL_MEASURES_ARGUMENT_LONG_OPTION                     : str       = "--all-measurements"
    C_ALL_MEASURES_ARGUMENT_DESCRIPTION                     : str       = "show all measures computed by NVIDIA scan tool"

    # recolect metrics
    C_GRAPH_ARGUMENT_SHORT_OPTION                           : str       = "-g"
    C_GRAPH_ARGUMENT_LONG_OPTION                            : str       = "--graph"
    C_GRAPH_ARGUMENT_DESCRIPTION                            : str       = "show graph with description of results"
    
    C_INTRO_MESSAGE_GENERAL_INFORMATION                 : str       = ("\n- Program Name:    topdown.py\n" + \
                                                                    "- Author:          Alvaro Saiz (UC)\n" + \
                                                                    "- Contact info:    asf174@alumnos.unican.es\n" + \
                                                                    "- Company:         University Of Cantabria\n" + \
                                                                    "- Place:           Santander, Cantabria, Kingdom of Spain\n" + \
                                                                    "- Advisors:        Pablo Abad (UC) <pablo.abad@unican.es>, Pablo Prieto (UC) <pablo.prieto@unican.es>\n" + \
                                                                    "- Bugs Report:     asf174@alumnos.unican.es"+ 
                                                                    "\n\n- Licence:         GNU GPL")

    C_NUM_MAX_CHARACTERS_PER_LINE                       : int       = 129
    C_MAX_NUM_RESULTS_DECIMALS                          : int       = 3


    C_COMPUTE_CAPABILITY_MAX_VALUE                      : float     = 8.0
    C_COMPUTE_CAPABILITY_NVPROF_MAX_VALUE               : float     = 6.2

