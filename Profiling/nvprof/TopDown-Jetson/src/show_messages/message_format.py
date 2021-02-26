"""
Program that shows messages in different format.

@author:    Alvaro Saiz (UC)
@date:      Jan-2021
@version:   1.0
"""
## faltan comentarios y comparar diferencia str None, o ""
import textwrap # text message
import sys
path : str = "/home/alvaro/Documents/Facultad/"
path_desp : str = "/mnt/HDD/alvaro/"
sys.path.insert(1, path + "TopDownNvidia/Profiling/nvprof/TopDown-Jetson/src/errors")
from errors.message_format_errors import *

class MessageFormat:
    """Class with different methods to show messages."""
    
    def __write_str_in_file(self, str_to_write : str, output_file : str):
        """ Write string ((if it's correct) in file.

        Params:
            str_to_write    : str   ; string to write
            output_file     : str   ; path to file
        """
        if output_file != "" or not output_file is None:
            try:
                f : _io.TextIOWrapper = open(output_file, "a")
                try:
                    #f.write("\n".join(str(item) for item in message))
                    f.write(str_to_write)

                finally:
                    f.close()
            except:  
                raise WriteInOutPutFileError
        pass

    def print_msg_box(self, msg, indent, width, title, output_file : str):
        """Print message-box with optional title."""

        lines = msg.split('\n')
        space = " " * indent
        if not width:
            width = max(map(len, lines))
        box = f'╔{"═" * (width + indent * 2)}╗\n'  # upper_border
        if title:
            box += f'║{space}{title:<{width}}{space}║\n'  # title
            box += f'║{space}{"-" * len(title):<{width}}{space}║\n'  # underscore
        box += ''.join([f'║{space}{line:<{width}}{space}║\n' for line in lines])
        box += f'╚{"═" * (width + indent * 2)}╝'  # lower_border
        print(box)
        self.__write_str_in_file(box, output_file)
        pass

    def print_center_msg_box(self, msg, indent, width, title, output_file : str):
        """Print message-box with optional title."""

        lines = msg.split('\n')
        space = " " * indent
        if not width:
            width = max(map(len, lines))
        box = f'\t\t\t\t\t\t\t╔{"═" * (width + indent * 2)}╗\n'  # upper_border
        if title:
            box += f'\t\t\t\t\t\t\t║{space}{title:<{width}}{space}║\n'  # title
            box += f'\t\t\t\t\t\t\t║{space}{"-" * len(title):<{width}}{space}║\n'  # underscore
        box += ''.join([f'\t\t\t\t\t\t\t║{space}{line:<{width}}{space}║\n' for line in lines])
        box += f'\t\t\t\t\t\t\t╚{"═" * (width + indent * 2)}╝'  # lower_border
        print(box)
        self.__write_str_in_file(box, output_file)
        pass

    def print_desplazed_msg_box(self, msg, indent, width, title, output_file : str):
        """Print message-box with optional title."""

        lines = msg.split('\n')
        space = " " * indent
        if not width:
            width = max(map(len, lines))
        box = f'\t\t\t╔{"═" * (width + indent * 2)}╗\n'  # upper_border
        if title:
            box += f'\t\t\t║{space}{title:<{width}}{space}║\n'  # title
            box += f'\t\t\t║{space}{"-" * len(title):<{width}}{space}║\n'  # underscore
        box += ''.join([f'\t\t\t║{space}{line:<{width}}{space}║\n' for line in lines])
        box += f'\t\t\t╚{"═" * (width + indent * 2)}╝'  # lower_border
        print(box)
        self.__write_str_in_file(box, output_file)
        pass

    def print_max_line_length_message(self, message : str, max_length : int, output_file : str):
        """Print Message with max length per line."""

        print('\n'.join(textwrap.wrap(message, max_length, break_long_words = False)))
        self.__write_str_in_file(message, output_file)
        pass

    def write_in_file_at_end(self, file : str, message : list[str]):
        """
        Write 'message' at the end of file with path 'file'

        Params:
            file    : str           ; path to file to write.
            message : list[str]     ; list of string with the information to be written (in order) to file.
                                      Each element of the list corresponds to a line.

        Raises:
            WriteInOutPutFileError  ; error when opening or write in file. Operation not performed
        """

        try:
            f : _io.TextIOWrapper = open(file, "a")
            try:
                f.write("\n".join(str(item) for item in message))

            finally:
                f.close()
        except:  
            raise WriteInOutPutFileError
        pass