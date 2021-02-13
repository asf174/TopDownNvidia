import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, "../src/")

from shell import Shell # launch shell arguments

bash = Shell()
command : str = "python3.9 ../src/topdown.py -f ../../../../CUDA/bin/add_two_matrix2 -o ../results/output.log -l1 -ld"
output : str = bash.launch_command_show_error(command, None)
print(output)