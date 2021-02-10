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
            sh.run(args = command, shell = True, executable = '/bin/bash')
            return True
        return False
    
   
    def launch_command_redirect(self, command : str, message : str, dest : str, add_to_end_file : bool) -> str :
        """
        Launch Shell command and redirect all output to 'dest' file.

        Params:
            command: str, command to launch in shell
            message: str, information about command. 
                        'None' to show no message
            dest: str, path to dest file.

        Returns:
            String with the output of command, or 'None' if an error ocurred and
            and the information could not be stored in the file or the file could 
            not be opened.
        
            If 'None' if returned there is no guarantee that the command has been 
            executed.
        """

        str_output : str = None
        try:
            open_mode : str = "a" # set as end by default
            if not add_to_end_file:
                open_mode = "w"
            f : _io.TextIOWrapper = open(dest, open_mode)
            try:
                if command:
                    self.__check_and_show_message(message)
                    output : sh.CompletedProcess = sh.run(args = command, shell = True, check = True, 
                        stdout = sh.PIPE, stderr = sh.STDOUT, text = True, executable = '/bin/bash') # text to use as string
                    f.write(output.stdout)
                    str_output = output.stdout
            finally:
                f.close()
        except:  
            pass # No need to do nothing, only don't execute command
        return str_output
    pass

#shell = Shell()
#returna : str = shell.launch_command_redirect("ls -l", None, "file", True)
#print(returna)