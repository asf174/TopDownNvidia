import os
import sys


class Shell_Commands:
    def _argumentsAreCorrect(self):
        
        # Arguments features
        MIN_NUMBER_ARGUMENTS = 3
        MAX_NUMBER_ARGUMENTS = MIN_NUMBER_ARGUMENTS
        HELP_OPTION = "help"
        HELP_OPTION_UPPER_ALL = "HELP"
        HELP_OPTION_UPPER = "Help"
        
        if len(sys.argv) < MIN_NUMBER_ARGUMENTS or len(sys.argv) > MAX_NUMBER_ARGUMENTS:
            if len(sys.argv) == 1:
                print("Error with number arguments")
            elif sys.argv[1] == HELP_OPTION or sys.argv[1] == HELP_OPTION_UPPER_ALL or sys.argv[1] == HELP_OPTION_UPPER:
                print("python3 " + sys.argv[0] + " <pathPrograma> <opciones>")
            return False
        return True
    pass
    def nvprof(self):
        if not self._argumentsAreCorrect():
            sys.exit()
        os.system("sudo $(which nvprof) " + sys.argv[1] + " " + sys.argv[2])
    
    def main(self):
        self.nvprof()
    pass

# execute 
Shell_Commands().main()

