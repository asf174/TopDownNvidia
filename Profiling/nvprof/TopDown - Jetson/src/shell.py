"""
Program that launch Shell commands.

@author:    Alvaro Saiz (UC)
@date:      Jan 2021
@version:   1.0
"""

"""To Launch Shell commands."""
import subprocess as sh

class Shell:
    """
    Class that launch shell commands with information.
    """
    
    def __check_and_show_message(self, message : str):
        """
        Show non-empty message. Don't show if 'message' is empty.

        Params:
             message: str, message to be showed.
                            'None' to show no message
        """

        if message:
            print(message)
    pass

    def launch_command(self, command: str, message : str) -> bool:
        """
        Launch Shell command.
        
        Params:
            command: str, command to launch in shell
            message: str, information about command.
                            'None' to show no message
        Returns:
            True if the command is executed correctly, or False 
            if the command is empty
        """
        
        # check empty command
        if command:
            self.__check_and_show_message(message)
            sh.call(command, shell = True, executable = '/bin/bash')
            return True
        return False
    
   
    def launch_command_redirect(self, command : str, message : str, dest : str, add_to_end_file : bool) -> bool :
        """
        Launch Shell command and redirect all output to 'dest' file.

        Params:
            command: str, command to launch in shell
            message: str, information about command. 
                        'None' to show no message
            dest: str, path to dest file.

        Returns:
            True if the command is executed correctly, or False 
            if the command is empty or 'dest' file cannot be opened
            to write
        """

        is_correct : bool = False
        try:
            open_mode : str = "a" # set as end by default
            if not add_to_end_file:
                open_mode = "w"
            f = open(dest, open_mode)
            try:
                if command:
                    self.__check_and_show_message(message)
                    output : sh.CompletedProcess = sh.run(args = command, shell = True, check = True, 
                        stdout = sh.PIPE, stderr = sh.STDOUT, text = True) # text to use as string
                    f.write(output.stdout)
                    is_correct = True
            finally:
                f.close()
        except IOError:  
            pass # No need to do nothing, only don't execute command
        return is_correct
    pass

#shell = Shell()
#returna : bool = shell.launch_command_redirect("ls -l", None, "/home/")
#print( str(returna))