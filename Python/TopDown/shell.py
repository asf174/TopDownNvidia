import subprocess as sh

class Shell:
    def launch_command(self, command: str) -> bool:
        # check empty command
        if command:
            sh.call(command, shell=True)
            return True
        return False
