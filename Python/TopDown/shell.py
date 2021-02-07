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
            if message:
                print(message)
            sh.call(command, shell=True)
            return True
        return False
