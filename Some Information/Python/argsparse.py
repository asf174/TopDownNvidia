import argparse
parser = argparse.ArgumentParser()

parser : argparse.ArgumentParser = argparse.ArgumentParser()
print(type(parser))
"""parser.add_argument (
    '-l', '--level',
    help = 'performs temperature test (period in sec)',
    type = int,
    nargs= '?',
    default = -1,
    #choices = range(1,3), # range [1,2], produces error, no if needed
    metavar= 'NUM',
    dest = 'level')   
args = parser.parse_args()
if args.level not in range(1, 2):
    print("OUT RANGO")
else:
    print("EN RANGO")"""

"""parser.add_argument(
    '-o', 
    '--outfile', 
    help= 'output file. Path to file',
    default = None,
    nargs='?', 
    type = argparse.FileType('w'), 
    metavar='/path/to/file',
    dest = 'file')
args = parser.parse_args()
args.file.write("HOLAAAAAAAAAAA")
args.file.close()"""

parser.add_argument (
    '-ld', 
    '--long-desc', 
    help= 'long description of results',
    action='store_true',
    dest = 'desc')
args : argparse.Namespace = parser.parse_args()
print(type(args))
if args.desc:
    print("HAY LARGA DESCRIPCION")
print("Value: " + str(args.desc))
