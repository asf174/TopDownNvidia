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

def print_four_msg_box(msgs, indent, titles):
        """Print message-box with optional title."""
        width1 = len(msgs[0][0])
        width2 = len(msgs[0][1])
        width3 = len(msgs[0][2])
        width4 = len(msgs[0][3])
        for i in range(1, len(msgs)):
            if width1 < len(msgs[i][0]):
                width1 = len(msgs[i][0])
            if width2 < len(msgs[i][1]):
                width2 = len(msgs[i][1])
            if width3 < len(msgs[i][2]):
                width3 = len(msgs[i][2])
            if width4 < len(msgs[i][3]):
                width4 = len(msgs[1][3])
    
        space = " " * indent
        box = f'╔{"═" * (width1 + indent * 2)}╗  '  # upper_border
        box += f'╔{"═" * (width2 + indent * 2)}╗  '  # upper_border
        box += f'╔{"═" * (width3 + indent * 2)}╗  '  # upper_border
        box += f'╔{"═" * (width4 + indent * 2)}╗\n'
        if titles:
            box += f'║{space}{titles[0]:<{width1}}{space}║  '  # title
            box += f'║{space}{titles[1]:<{width2}}{space}║  '  # title
            box += f'║{space}{titles[2]:<{width3}}{space}║  '  # title
            box += f'║{space}{titles[3]:<{width4}}{space}║\n'  # title
            box += f'║{space}{"-" * len(titles[0]):<{width1}}{space}║  '  # underscore
            box += f'║{space}{"-" * len(titles[1]):<{width2}}{space}║  '  # underscore
            box += f'║{space}{"-" * len(titles[2]):<{width3}}{space}║  '  # underscore  
            box += f'║{space}{"-" * len(titles[3]):<{width4}}{space}║\n'  # underscore
        
        for i in range(0, len(msgs)):
            box += ''.join(f'║{space}{msgs[i][0]:<{width1}}{space}║  ')
            box += ''.join(f'║{space}{msgs[i][1]:<{width2}}{space}║  ')
            box += ''.join(f'║{space}{msgs[i][2]:<{width3}}{space}║  ')
            box += ''.join(f'║{space}{msgs[i][3]:<{width4}}{space}║  ')
            box += "\n"
        box += f'╚{"═" * (width1 + indent * 2)}╝  '  # lower_border
        box += f'╚{"═" * (width2 + indent * 2)}╝  '  # lower_border
        box += f'╚{"═" * (width3 + indent * 2)}╝  '  # lower_border
        box += f'╚{"═" * (width4 + indent * 2)}╝  '  # lower_border
        print(box)
        pass


def print_n_per_line_msg_box(matrix : list[list], titles, indent, width):
        """Print message-box with optional title."""

        lines = matrix[0][0].split('\n') # by default
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
                    box += f'║{space}{"-" * len(titles[i]):<{width}}{space}║\n'  # underscore
                else:
                    box += f'║{space}{"-" * len(titles[i]):<{width}}{space}║  '  # underscore
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
        for i in range(np.size(matrix, 1)):
            box += f'╚{"═" * (width + indent * 2)}╝  '  # lower_border
        print(box)
        pass

#me = ("{:<25} {:<2}".format('STALLS, on the total (%): ', str(1) + '%'))
#print(len("STALLS, on the total (%):"))
#message = [ [me,me,me,me], [" "," "," "," "]]
#titles = ["FRONT END", "BACK END", "DIVERGENCE", "RETIRE"]
#message += ("{:<5} {:<5}".format('IPC DEGRADATION (%):', str(1) + '%'))

#message = ("STALLS, on the total (%):          str(1) + '%\n\n'))
#message += ("{:<20} {:<5}".format('IPC DEGRADATION (%):', str(1) + '%\n'))

#stri : str = print_4_msg_box(msg = "STALLS, on the total (%):", msg2 = "STALLS, on the total (%):", msg3 = "STALLS, on the total (%):", msg4 = "STALLS, on the total (%):",
# indent = 1, title = "RESULTS TOP", width = None)
#print_n_per_line_msg_box(matrix = message, titles = titles, indent = 1, width = None )


### prueba de los 4
msgs = [["HOLA alvaro, prueba","HOLA 2222222","HOLA 33333","HOLA sadasds"], 
        ["","HOLA 2222222","HOLA PRUEBA","HOLA sadasds"],
        ["PABLO","","",""]]
titles = ["FRONT END", "BACK END", "DIVERGENCE", "RETIRE"]
print_four_msg_box(msgs, 1, titles)