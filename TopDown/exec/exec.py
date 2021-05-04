
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, "../src/shell")

from shell import Shell # launch shell arguments

bash = Shell()
command : str = "python3 ../src/topdown.py  -f ../../CUDA/bin/add_two_matrix -l2 -o ../results/file.log -am -nd -g"
output : str = bash.launch_command_show_all(command, None)
print(output)
