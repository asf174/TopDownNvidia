import textwrap

lorem = """Lorem ipsum dolor sit amet, consectetur adipiscing elit. 
Phasellus ac commodo libero, at dictum leo. Nunc convallis est id purus porta,  
malesuada erat volutpat. Cras commodo odio nulla. Nam vehicula risus id lacus 
vestibulum. Maecenas aliquet iaculis dignissim. Phasellus aliquam facilisis  
pellentesque ultricies. Vestibulum dapibus quam leo, sed massa ornare eget. 
Praesent euismod ac nulla in lobortis. 
Sed sodales tellus non semper feugiat."""

def wrapped_lines(line, width=80):
    whitespace = set(" \n\t\r")
    length = len(line)
    start = 0

    while start < (length - width):
        # we take next 'width' of characters:
        chunk = line[start:start+width+1]
        # if there is a newline in it, let's return first part
        if '\n' in chunk:
            end = start + chunk.find('\n')
            yield line[start:end]
            start = end+1 # we set new start on place where we are now
            continue

        # if no newline in chunk, let's find the first whitespace from the end
        for i, ch in enumerate(reversed(chunk)):
            if ch in whitespace:
                end = (start+width-i)
                yield line[start:end]
                start = end + 1
                break
            else: # just for readability
                continue 
    yield line[start:]

for line in wrapped_lines(lorem, 30):
    print (line)


import textwrap
txt = "Lorem ipsum dolor sit amet, consectetur adipiscing elit."
print ('\n'.join(textwrap.wrap(txt, 100, break_long_words=False)))


def print_msg_box(msg, indent=1, width=None, title=None):
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
    return box

stri : str = print_msg_box(msg = "       Hola", indent = 1, title = "RESULTS")
str2 :str = print_msg_box(msg = "Pruebs", indent = 1, title = "RESULTS")
print("\t\t\t{:<5} {:<4} ".format(stri,str2))