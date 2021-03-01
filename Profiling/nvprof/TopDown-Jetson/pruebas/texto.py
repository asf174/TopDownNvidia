import textwrap
import numpy as np




def print_2_msg_box(msg, msg2, indent, width, title):
        """Print message-box with optional title."""

        lines = msg.split('\n')
        space = " " * indent
        if not width:
            width = max(map(len, lines))
        box = f'╔{"═" * (width + indent * 2)}╗  '  # upper_border
        box += f'╔{"═" * (width + indent * 2)}╗\n'
        if title:
            box += f'║{space}{title:<{width}}{space}║  '  # title
            box += f'║{space}{title:<{width}}{space}║\n'  # title
            box += f'║{space}{"-" * len(title):<{width}}{space}║  '  # underscore
            box += f'║{space}{"-" * len(title):<{width}}{space}║\n'  # underscore

        box += ''.join([f'║{space}{line:<{width}}{space}║  ' for line in lines])
        box += ''.join([f'║{space}{line:<{width}}{space}║\n' for line in lines])
        box += f'╚{"═" * (width + indent * 2)}╝  '  # lower_border
        box += f'╚{"═" * (width + indent * 2)}╝  '  # lower_border
        print(box)
        pass

def print_4_msg_box(matrix, titles, indent, width):
        """Print message-box with optional title."""

        lines = matrix[0][0].split('\n')
        space = " " * indent
        if not width:
            width = max(map(len, lines))
        box = ""
        for i in range(len(titles) - 1):
            box += f'╔{"═" * (width + indent * 2)}╗  '  # upper_border
        box += f'╔{"═" * (width + indent * 2)}╗\n'
        if len(titles) > 1:
            for i in range(len(titles)):
                if i == len(titles) - 1:
                    box += f'║{space}{titles[i]:<{width}}{space}║\n'  # title
                else:
                    box += f'║{space}{titles[i]:<{width}}{space}║  '  # title
            for i in range(len(titles)):
                if i == len(titles) - 1:
                    box += f'║{space}{"-" * len(titles):<{width}}{space}║\n'  # underscore
                else:
                    box += f'║{space}{"-" * len(titles):<{width}}{space}║  '  # underscore
        num_ite : int = 0
        for i in range(len(matrix)):
            if i == len(matrix) - 1:
                box += "\n"
            for j in range(len(matrix[i])):
                if num_ite ==  len(matrix)*len(matrix[0])- 1 :
                    box += f'║{space}{matrix[i][j]:<{width}}{space}║\n'
                else:
                    box += f'║{space}{matrix[i][j]:<{width}}{space}║  '
            num_ite = num_ite + 1
        box += "\n"
        print(num_ite)
        for i in range(np.size(matrix, 1)):
            box += f'╚{"═" * (width + indent * 2)}╝  '  # lower_border
        print(box)
        pass

#message = ("{:<5} {:<5}".format('\nSTALLS, on the total (%):', str(1) + '%'))
message = [ ["Hola afkjlsd","SALUDOS1","SALUDOS2","SALUDOS3"], [" "," "," "," "]]
titles = ["FRONT END", "BACK END", "DIVERGENCE", "RETIRE"]
#message += ("{:<5} {:<5}".format('IPC DEGRADATION (%):', str(1) + '%'))

#message = ("STALLS, on the total (%):          str(1) + '%\n\n'))
#message += ("{:<20} {:<5}".format('IPC DEGRADATION (%):', str(1) + '%\n'))

#stri : str = print_4_msg_box(msg = "STALLS, on the total (%):", msg2 = "STALLS, on the total (%):", msg3 = "STALLS, on the total (%):", msg4 = "STALLS, on the total (%):",
# indent = 1, title = "RESULTS TOP", width = None)
print_4_msg_box(matrix = message, titles = titles, indent = 1, width = None )