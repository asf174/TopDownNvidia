from pathlib import Path
import re

results_launch = Path("file.txt").read_text()
for line in results_launch.splitlines():
    line = re.sub(' +', ' ', line) # delete more than one spaces and put only one
    list_words = line.split(" ")
    print(list_words)       
