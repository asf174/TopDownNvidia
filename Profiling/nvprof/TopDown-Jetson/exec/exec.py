import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, "../src/shell")

from shell import Shell # launch shell arguments

bash = Shell()
<<<<<<< HEAD
command : str = "python3.9 ../src/topdown.py -f ../../../../CUDA/bin/add_two_matrix -o ../results/output.log -l2 -o pene"
=======
command : str = "python3.9 ../src/topdown.py -f ../../../../CUDA/bin/add_two_matrix -o ../results/output.log -l2 -ld"
>>>>>>> e6c4273bdfa2f5e7a61463c6237948da7a962a74
output : str = bash.launch_command_show_all(command, None)
print(output)