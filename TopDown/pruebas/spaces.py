import re
s = "          1                          stall_inst_fetch     Issue Stall Reasons (Instructions Fetch)       3.22%       3.22%       3.22%"
s = re.sub(' +', ' ', s)
print(s)

list_words = s.split(" ")
print(list_words)
if list_words[0] == '' and list_words[len(list_words) - 1 ][0].isnumeric():
    print("CORRECTO")
print(list_words[len(list_words) - 1 ][0])