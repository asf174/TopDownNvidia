"""
Program that implements the Top Down methodology on GPUs.

@author:    Alvaro Saiz (UC)
@date:      Jan 2021
@version:   1.0
"""
import sys # for arguments
from shell import Shell # launch shell arguments

class TopDown():
    """Class that implements TopDown methodology."""
    
    # Arguments features
    C_MIN_NUMBER_ARGUMENTS              : int       = 1
    C_MAX_NUMBER_ARGUMENTS              : int       = 100*C_MIN_NUMBER_ARGUMENTS
    
    """
    Options of program
    """
    # Level
    C_LEVEL_1_SHORT_OPTION              : str       = "-l1"
    C_LEVEL_2_SHORT_OPTION              : str       = "-l2"
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
    
    def read_arguments(self) -> bool:   
        """
        Check if arguments passed to program are correct. 
        Show information with '-h'/'--help' option

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

    
    def find_level(self) -> int:
        """ 
        Find the TopDown run level

        Returns:
            1 if it's level one, two if it's level two
            or -1 to nofity ERROR (no level specified in the correct 
            way)
        """ 

        for i in range(1, len(sys.argv)):
            # check simple arguments
            if sys.argv[i] == self.C_LEVEL_1_SHORT_OPTION:
                return 1
            if sys.argv[i] == self.C_LEVEL_2_SHORT_OPTION:
                return 2

            # check long arguments
            if i + 1 <= len(sys.argv):
                if sys.argv[i] == self.C_LEVEL_LONG_OPTION:
                    if sys.argv[i+1] == "1":
                        return 1
                    if sys.argv[i+1] == "2":
                        return 2
        return -1
    pass

    def find_output_file(self) -> str:
        """
        Find path to output file

        Returns:
            string with path to file, or None if 
            option '-o' or '--output' has not been indicated
        """

        for i in range(1, len(sys.argv)):
            # check long arguments
            if i + 1 <= len(sys.argv):
                if sys.argv[i] == self.C_OUTPUT_FILE_SHORT_OPTION or sys.argv[i] == self.C_OUTPUT_FILE_LONG_OPTION:
                    return sys.argv[i + 1]
        return None
    pass
    
    def show_long_desc(self) -> bool:
        """
        Check if program has to show long description of results.

        Returns:
            True to show long description of False if not
        """

        for i in range(1, len(sys.argv)):
            if (sys.argv[i] == self.C_LONG_DESCRIPTION_SHORT_OPTION or 
                sys.argv[i] == self.C_LONG_DESCRIPTION_LONG_OPTION):
                return True
        return False
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
    level : int = td.find_level()
    
    if level == -1:
        print("ERROR LEVEL")
        sys.exit()
    print("LEVEL: " + str(level))

    file : str = td.find_output_file()
    if file is None:
        print("NO OUTPUT-FILE")
    else:
        print("OUTPUT-FILE: " + file)

    showLongDesc : bool = td.show_long_desc()
    if showLongDesc:
        print("SHOW Long-Desc")
    else:
        print ("DO NOT SHOW Long-Desc")

    if level == 1:
        td.level_1()
    else:
        td.level_2()