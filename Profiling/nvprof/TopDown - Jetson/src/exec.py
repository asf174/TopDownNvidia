from shell import Shell # launch shell arguments

bash = Shell()
command : str = "python3.9 topdown.py -f ../../../../CUDA/bin/add_two_matrix2 -o output.txt -l1"
bash.launch_command(command, None) 