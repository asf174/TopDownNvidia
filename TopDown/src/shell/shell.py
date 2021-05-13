"""
Program that launch Shell commands.

@author:    Alvaro Saiz (UC)
@date:      Jan 2021
@version:   1.0
"""

import subprocess as sh # To Launch Shell commands.
import os

class Shell:
    """
    Class that launch shell commands with information.
    """
     
    def __launch_shell_and_message(self, command: str, message : str, hasToCheck : bool) -> str:
        """
        Launch Shell command.
        
        Params:
            command     : str   ; command to launch in shell
            message     : str   ; information about command.
                                  'None' to show no message
            hasToCheck  : bool  ; True to do not show the execution of command if command produces errors 
                                  or false in other case 
        Returns:
            String with all the output of command, or 'None' if an error ocurred
            launching command
        """

        str_output : str = None
        try:
            if command:
                if message:
                    print(message)
                output : sh.CompletedProcess = sh.run(args = command, shell = True, check = hasToCheck, 
                stdout = sh.PIPE, stderr = sh.STDOUT, executable = '/bin/bash', env = dict(os.environ)) # text to use as string
                str_output = output.stdout.decode('utf-8')
        except : 
            pass # No need to do nothing, command was not executed succesfully
        return str(str_output)
        pass
    
    def launch_command(self, command: str, message : str) -> str:
        """
        Launch Shell command.
        
        Params:
            command     : str   ; command to launch in shell
            message     : str   ; information about command.
                                  'None' to show no message
        Returns:
            String with all the output of command, or 'None' if an error ocurred
            launching command
        """

        return self.__launch_shell_and_message(command, message, True)
        pass
    
    def launch_command_redirect(self, command : str, message : str, dest : str, add_to_end_file : bool) -> str :
        """
        Launch Shell command and redirect output to 'dest' file 
        in case the command is executed correctly.

        Params:
            command             : str   ; command to launch in shell
            message             : str   ; information about command.
                                          'None' to show no message
            dest                : str   ; path to dest file.
            add_to_end_file     : bool  ; True to add 'message' to the end of 'dest' file
                                          or False if not
        Returns:
            String with all the output of command, or 'None' if an error ocurred and
            and the information could not be stored in the file or the file could 
            not be opened or error during command execution
        """

        str_output : str = None
        try:
            str_output = self.__launch_shell_and_message(command, message, True)
            if not str_output is None:
                open_mode : str = "a" # set as end by default
                if not add_to_end_file:
                    open_mode = "w"
                f : _io.TextIOWrapper = open(dest, open_mode)
                try:
                    f.write(str_output)
                finally:
                    f.close()
        except:  
            str_output = None
        return str_output
        pass

    def launch_command_show_all(self, command: str, message : str) -> str:
        """
        Launch Shell command and show the result of the execution (including errors)
        
        Params:
            command     : str   ; command to launch in shell
            message     : str   ; information about command.
                                  'None' to show no message

        Returns:
            String with the output of the command executed
        """
        return self.__launch_shell_and_message(command, message, False)
        pass
