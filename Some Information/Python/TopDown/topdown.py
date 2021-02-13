"""
Program that implements the Top Down methodology on GPUs.

@author:    Alvaro Saiz (UC)
@date:      Jan 2021
@version:   1.0
"""
import sys # for arguments
from shell import Shell # launch shell arguments
import argparse

class TopDown():
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
    C_LEVEL_1_FRONT_END_METRIS          : str       = "stall_inst_fetch, stall_exec_dependency, stall_sync, stall_other, stall_not_selected"
    C_LEVEL_1_BACK_END_METRIS           : str       = ""

    def __init__(self):
        parser : argparse.ArgumentParse = argparse.ArgumentParser(prog='[/path/to/PROGRAM]',
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
    
    def __add_program_argument(self, requiredGroup):
        requiredGroup.add_argument(
            self.C_FILE_SHORT_OPTION, 
            self.C_FILE_LONG_OPTION, 
            help = 'run file. Path to file.',
            default = None,
            nargs = '?', 
            type = argparse.FileType('w+'), 
            #metavar='/path/to/file',
            dest = 'program')
    pass

    def __add_level_argument(self, requiredGroup):
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
            type = argparse.FileType('w+'), 
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
        print("Llego aqui")
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

    
    def level(self) -> int:
        """ 
        Find the TopDown run level

        Returns:
            1 if it's level one, 2 if it's level two
        """ 
        return self.__level
    pass

    def output_file(self):
        """
        Find path to output file

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

    
    
    
    
    def level_1(self):
        """ 
        Run TopDown level 1.
        """

        shell : Shell = Shell()
        # FrontEnd Commands
        command : str = ("sudo $(which nvprof) --metrics " + self.C_LEVEL_1_FRONT_END_METRIS + 
            " --unified-memory-profiling off --profile-from-start off " + self.__program)
        shell.launch_command(command, "LAUNCH FRONT END")
    pass

    def level_2(self):
        """ 
        Run TopDown level 2.
        """
        
        # TODO
        shell = Shell()
        shell.launch_command("ls", "LAUNCH ls")
        return True
    pass
    
if __name__ == '__main__':
    td = TopDown()
    
    if not td.read_arguments():
        sys.exit()
    level : int = td.level()
    
    if level == -1:
        print("ERROR LEVEL")
        sys.exit()
    print("LEVEL: " + str(level))

    file : str = td.output_file()
    if file is None:
        print("NO OUTPUT-FILE")
    else:
        print("OUTPUT-FILE: ")

    showLongDesc : bool = td.show_long_desc()
    if showLongDesc:
        print("SHOW Long-Desc")
    else:
        print ("DO NOT SHOW Long-Desc")

    """if level == 1:
        td.level_1()
    else:
        td.level_2()"""