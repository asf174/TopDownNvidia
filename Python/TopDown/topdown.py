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

    """Options."""
    C_MAX_LEVEL_EXECUTION               : int       = 2
    C_MIN_LEVEL_EXECUTION               : int       = 1
    
    def __init__(self):
        #self.__parser : argparse.ArgumentParser = argparse.ArgumentParser(description='Process some integers.') # Arguments
        self.__parser = argparse.ArgumentParser(prog='tool',
            formatter_class=lambda prog: argparse.HelpFormatter(prog,max_help_position = 50),
            description = "TopDown methodology on GPU",
            epilog="Check options to run program") #exit_on_error=False)
        self.__parser._optionals.title = "Optional Arguments"
        self.__add_arguments()

        # Save values into attributes
        args : argparse.Namespace = self.__parser.parse_args()

        self.__level : int = args.level[0]
        self.__file = args.file
        self.__show_long_desc : bool = args.desc
    pass
    

    def __add_level_argument(self):
        requiredNamed = self.__parser.add_argument_group("Required arguments")
        requiredNamed.add_argument (
            self.C_LEVEL_SHORT_OPTION, self.C_LEVEL_LONG_OPTION,
            required = True,
            help = 'level of execution',
            type = int,
            nargs = 1,
            default = -1,
            choices = range(1,3), # range [1,2], produces error, no if needed
            metavar = 'NUM',
            dest = 'level')
    pass

    def __add_ouput_file_argument(self):
        self.__parser.add_argument(
            self.C_OUTPUT_FILE_SHORT_OPTION, 
            self.C_OUTPUT_FILE_LONG_OPTION, 
            help = 'output file. Path to file.',
            default = None,
            nargs = '?', 
            type = argparse.FileType('w+'), 
            #metavar='/path/to/file',
            dest = 'file')
    pass

    def __add_long_desc_argument(self):
        self.__parser.add_argument (
            self.C_LONG_DESCRIPTION_SHORT_OPTION, 
            self.C_LONG_DESCRIPTION_LONG_OPTION, 
            help = 'long description of results',
            action = 'store_true',
            dest = 'desc')
    pass

    def __add_arguments(self):
        self.__add_level_argument()
        self.__add_ouput_file_argument()
        self.__add_long_desc_argument()
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
        
        # TODO
        shell = Shell()
        shell.launch_command("pwd", "LAUNCH PWD")
        return True
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