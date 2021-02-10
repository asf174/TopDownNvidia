"""
Program that implements the Top Down methodology on GPUs.

@author:    Alvaro Saiz (UC)
@date:      Jan 2021
@version:   1.0
"""
import sys # for arguments
from shell import Shell # launch shell arguments
import argparse
import re
#from tabulate import tabulate #TODO, pintar

from errors import FrontAndBackErrorOperation, WriteInOutPutFileError

class TopDown:
    """Class that implements TopDown methodology."""
    
    # Arguments features
    C_MIN_NUMBER_ARGUMENTS              : int       = 1
    C_MAX_NUMBER_ARGUMENTS              : int       = 100*C_MIN_NUMBER_ARGUMENTS
    
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
    C_MAX_LEVEL_EXECUTION               : int       = 2
    C_MIN_LEVEL_EXECUTION               : int       = 1
    
    """Metrics."""
    C_LEVEL_1_FRONT_END_METRICS         : str       = ("stall_inst_fetch,stall_exec_dependency,stall_sync,stall_other," +
                                                        "stall_not_selected,stall_not_selected")
    C_LEVEL_1_BACK_END_METRICS          : str       = ("stall_memory_dependency,stall_constant_memory_dependency,stall_pipe_busy," +
                                                        "stall_memory_throttle")
    C_LEVEL_1_DIVERGENCE_METRICS        : str       = ("")


    C_INFO_MESSAGE_EXECUTION_NVPROF     : str       = "Launching Command... Wait to results."

    def __init__(self):
        parser : argparse.ArgumentParse = argparse.ArgumentParser(#prog='[/path/to/PROGRAM]',
            formatter_class=lambda prog: argparse.HelpFormatter(prog,max_help_position = 50),
            description = "TopDown methodology on GPU",
            epilog="Check options to run program")
            #usage='%(prog)s [OPTIONS]') #exit_on_error=False)
        parser._optionals.title = "Optional Arguments"
        self.__add_arguments(parser)

        # Save values into attributes
        args : argparse.Namespace = parser.parse_args()
      
        self.__level : int = args.level[0]
        self.__file = args.file
        self.__show_long_desc : bool = args.desc
        self.__program = args.program
    pass
    
    def __add_program_argument(self, requiredGroup : argparse._ArgumentGroup):
        requiredGroup.add_argument(
            self.C_FILE_SHORT_OPTION, 
            self.C_FILE_LONG_OPTION, 
            required = True,
            help = 'run file. Path to file.',
            default = None,
            nargs = '?', 
            type = str, 
            #metavar='/path/to/file',
            dest = 'program')
    pass

    def __add_level_argument(self, requiredGroup : argparse._ArgumentGroup):
        requiredGroup.add_argument (
            self.C_LEVEL_SHORT_OPTION, self.C_LEVEL_LONG_OPTION,
            required = True,
            help = 'level of execution.',
            type = int,
            nargs = 1,
            default = -1,
            choices = range(self.C_MIN_LEVEL_EXECUTION, self.C_MAX_LEVEL_EXECUTION + 1), # range [1,2], produces error, no if needed
            metavar = '[NUM]',
            dest = 'level')
    pass

    def __add_ouput_file_argument(self, parser : argparse.ArgumentParser):
        parser.add_argument (
            self.C_OUTPUT_FILE_SHORT_OPTION, 
            self.C_OUTPUT_FILE_LONG_OPTION, 
            help = 'output file. Path to file.',
            default = None,
            nargs = '?', 
            type = str, 
            #metavar='/path/to/file',
            dest = 'file')
    pass

    def __add_long_desc_argument(self, parser : argparse.ArgumentParser):
        parser.add_argument (
            self.C_LONG_DESCRIPTION_SHORT_OPTION, 
            self.C_LONG_DESCRIPTION_LONG_OPTION, 
            help = 'long description of results.',
            action = 'store_true',
            dest = 'desc')
    pass
    

    def __add_arguments(self, parser : argparse.ArgumentParser):
        # Create group for required arguments
        requiredGroup : argparse._ArgumentGroup = parser.add_argument_group("Required arguments")

        self.__add_program_argument(requiredGroup)
        self.__add_level_argument(requiredGroup)
        self.__add_ouput_file_argument(parser)
        self.__add_long_desc_argument(parser)
    pass

    def read_arguments(self) -> bool:   
        """
        Check if arguments passed to program are correct. 
        Show information with '-h'/'--help' option. NOT USED

        Returns:
            True if are correct, False if not
        """
        if len(sys.argv) == self.C_MIN_NUMBER_ARGUMENTS:
            print("Error with arguments. Introduce '-h' or '--help' to see options")
            return False
        elif sys.argv[1] == self.C_HELP_SHORT_OPTION or sys.argv[1] == self.C_HELP_LONG_OPTION:
            print("NAME")
            print("\ttopdown - TopDown methodology on GPU\n")

            print("SYNOPSIS")
            print("\t"+ sys.argv[1] + " [OPTIONS] [/path/to/PROGRAM]\n")

            print("DESCRIPTION\n\n    List of Arguments:\n")
            print("\t-l, --level")
            print("\t\tIndicates the run level\n")
            
            print("\t-ld, --long-desc")
            print("\t\tShows a long description of the results\n")

            print("\t-h, --help")
            print("\t\tShows a description of program\n")

            print("\t-o, --output")
            print("\t\tSave results in file. Path to file")
            return False
        return True
    pass

    def program(self) -> str:
        """
        Returns path to runnable program.

        Returns:
            str with path to program to be executed
        """
        return self.__program
    pass
    
    def level(self) -> int:
        """ 
        Find the TopDown run level.

        Returns:
            1 if it's level one, 2 if it's level two
        """ 
        return self.__level
    pass

    def output_file(self):
        """
        Find path to output file.

        Returns:
            opened file to write, or None if 
            option '-o' or '--output' has not been indicated
        """
        return self.__file # descriptor to file or None

    pass
    
    def show_long_desc(self) -> bool:
        """
        Check if program has to show long description of results.

        Returns:
            True to show long description of False if not
        """
        return self.__show_long_desc
    pass

    def __write_in_file_at_end(self, file : str, message : list[str]):
        """
        CWrite 'message' at the end of file with path 'file'

        Params:
            file    : str       ; path to file to write.
            message : list[str] ; list of string with the information to be written (in order) to file.
                                  Each element of the list corresponds to a line.

        Raises:
            WriteInOutPutFileError ; error when opening or write in file. Operation not performed
        """

        try:
            f : _io.TextIOWrapper = open(file, "a")
            try:
                f.write("\n".join(str(item) for item in message))

            finally:
                f.close()
        except:  
            raise WriteInOutPutFileError # No need to do nothing, only don't execute command
    pass

    def level_1(self):
        """ 
        Run TopDown level 1.

        Raises:
            FrontAndBackErrorOperation ; raised in case of error reading results from NVPROF
        """

        shell : Shell = Shell()

        # FrontEnd Command
        command : str = ("sudo $(which nvprof) --metrics " + self.C_LEVEL_1_FRONT_END_METRICS + 
            "," + self.C_LEVEL_1_BACK_END_METRICS + "," + self.C_LEVEL_1_DIVERGENCE_METRICS 
            + " --unified-memory-profiling off --profile-from-start off " + self.__program)
        
        output_file : str = self.output_file()
        output_command : bool

        if output_file is None:
            output_command = shell.launch_command(command, self.C_INFO_MESSAGE_EXECUTION_NVPROF)
        else:
            output_command = shell.launch_command_redirect(command, self.C_INFO_MESSAGE_EXECUTION_NVPROF, output_file, True)

        if output_command is None:
            raise FrontAndBackErrorOperation
        else:
            # Create dictionaries with name of counters as key.
            if self.C_LEVEL_1_FRONT_END_METRICS != "":
                dictionary_front_counters : dict = dict.fromkeys(self.C_LEVEL_1_FRONT_END_METRICS.split(","))
                dictionary_front_counters_desc : dict = dict.fromkeys(self.C_LEVEL_1_FRONT_END_METRICS.split(","))
            if self.C_LEVEL_1_BACK_END_METRICS != "":
                dictionary_back_counters : dict = dict.fromkeys(self.C_LEVEL_1_BACK_END_METRICS.split(","))
                dictionary_back_counters_desc : dict = dict.fromkeys(self.C_LEVEL_1_BACK_END_METRICS.split(","))
            if self.C_LEVEL_1_DIVERGENCE_METRICS != "":
                dictionary_divergence_counters : dict = dict.fromkeys(self.C_LEVEL_1_DIVERGENCE_METRICS.split(","))
                dictionary_divergence_counters_desc : dict = dict.fromkeys(self.C_LEVEL_1_DIVERGENCE_METRICS.split(","))
            
            #print(output_command)
            for line in output_command.splitlines():
                line = re.sub(' +', ' ', line) # delete more than one spaces and put only one
                list_words = line.split(" ")
                
                # Check if it's line of interest:
                # ['', 'X', 'NAME_COUNTER', ... , 'Min', 'Max', 'Avg' (Y%)] -> X (int number), Y (int/float number)
                if list_words[0] == '' and list_words[len(list_words) - 1 ][0].isnumeric():
                    # CORRECT FORMAT: OUR LINE
                    name_counter : str = list_words[2]
                    description_counter : str = ""
                    for i in range(3, len(list_words) - 3):
                        description_counter += list_words[i] + " "     
                    avg_value : str = list_words[len(list_words) - 1]
                    max_value : str = list_words[len(list_words) - 2]
                    min_value : str = list_words[len(list_words) - 3]

                    #if avg_value != max_value or avg_value != min_value:
                        # Do Something. NOT USED

                    if name_counter in (dictionary_front_counters and dictionary_front_counters_desc) :
                        dictionary_front_counters[name_counter] = avg_value
                        dictionary_front_counters_desc[name_counter] = description_counter
                    
                    elif name_counter in (dictionary_back_counters and dictionary_back_counters_desc):
                        dictionary_back_counters[name_counter] = avg_value
                        dictionary_back_counters_desc[name_counter] = description_counter
                    
                    elif name_counter in (dictionary_divergence_counters and dictionary_divergence_counters_desc):
                        dictionary_divergence_counters[name_counter] = avg_value
                        dictionary_divergence_counters_desc[name_counter] = description_counter
                    else: # counter not defined
                        raise OutputResultCommandError

            
            #  Keep Results
            lst_output : list[str] = []
            lst_output.append("\nList of counters measured according to the part measured."
                                + " The result obtained is stored for each counter, as well as a description.")
            if self.C_LEVEL_1_FRONT_END_METRICS != "":
                lst_output.append("- FRONT-END RESULTS:")
                lst_output.append("\t\t\t{:<45} {:<49}  {:<5} ".format('Counter','Description', 'Value'))
                lst_output.append( "\t\t\t----------------------------------------------------"
                +"---------------------------------------------------")
                for (counter, value), (counter, desc) in zip(dictionary_front_counters.items(), # same key in both dictionaries
                    dictionary_front_counters_desc.items()):
                    lst_output.append("\t\t\t{:<45} {:<50} {:<6} ".format(counter, desc, value))

            if self.C_LEVEL_1_BACK_END_METRICS != "":
                lst_output.append( "\n\n- BACK-END RESULTS:")
                lst_output.append( "\t\t\t{:<45} {:<49}  {:<5} ".format('Counter','Description', 'Value'))
                lst_output.append( "\t\t\t----------------------------------------------------"
                +"---------------------------------------------------")
                for (counter,value), (counter,desc) in zip(dictionary_back_counters.items(), 
                dictionary_back_counters_desc.items()):
                    lst_output.append("\t\t\t{:<45} {:<50} {:<6} ".format(counter, desc, value))
                
            if self.C_LEVEL_1_DIVERGENCE_METRICS != "":
                lst_output.append("\n\n- DIVERGENCE RESULTS:")
                lst_output.append("\t\t\t{:<45} {:<49} {:<6}".format('Counter','Description', 'Value'))
                lst_output.append( "\t\t\t----------------------------------------------------"
                +"---------------------------------------------------")
                for (counter,value), (counter,desc) in zip(dictionary_divergence_counters.items(), 
                dictionary_divergence_counters_desc.items()):
                    lst_output.append("\t\t\t{:<45} {:<50} {:<6} ".format(counter, desc, value))
            
            if self.show_long_desc():
                # Write results in output-file if has been specified
                if not self.output_file() is None:
                    self.__write_in_file_at_end(self.output_file(),lst_output)
                for element in lst_output:
                    print(element)


    pass

    def level_2(self):
        """ 
        Run TopDown level 2.
        """
        
        # TODO
        shell = Shell()
        shell.launch_command("ls", "LAUNCH ls")
    pass
    
if __name__ == '__main__':
    td = TopDown()
    
   # if not td.read_arguments():
    #    sys.exit()
    #file : str = td.output_file()
    #if file is None:
    #    print("NO OUTPUT-FILE")
    #else:
    #    print("OUTPUT-FILE: " + file)

    #showLongDesc : bool = td.show_long_desc()
    #if showLongDesc:
    ##   print("SHOW Long-Desc")
    #else:
    #    print ("DO NOT SHOW Long-Desc")

    level : int = td.level()
    if level == -1:
        print("ERROR LEVEL")
        sys.exit()
    #print("LEVEL: " + str(level))
    if level == 1:
        td.level_1()
    elif level == 2:
        td.level_2()
    else:
        print("ERROR LEVEL")
        sys.exit()

