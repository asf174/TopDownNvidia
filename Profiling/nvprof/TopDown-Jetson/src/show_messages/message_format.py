"""
Program that shows messages in different format.

@author:    Alvaro Saiz (UC)
@date:      Jan-2021
@version:   1.0
"""

import textwrap # text message

class MessageFormat:
    """Class with different methods to show messages."""
    
    def print_msg_box(self, msg, indent = 1, width = None, title = None):
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
        pass

    def print_center_msg_box(self, msg, indent = 1, width = None, title = None):
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
        pass

    def print_desplazed_msg_box(self, msg, indent = 1, width = None, title = None):
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
        pass


    def print_max_line_length_message(self, message : str, max_length : int):
        """Print Message with max length per line."""

        print('\n'.join(textwrap.wrap(message, max_length, break_long_words = False)))
        pass