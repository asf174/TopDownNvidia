



a_dictionary = {"a": "16%", "b": "2%", "c": "12%"}
max = float('-inf')
print(max)
for key in a_dictionary.keys():
    print(str(a_dictionary.get(key)[0 : len(a_dictionary.get(key)) - 1]))

L = [('Sam', "35%"),
    ('Tom', "351%"),
    ('Bob', "350%")]

#x = max(a_dictionary, key=myFunc)
#print(x)
# Prints ('Sam', 35)

