import sys
def printf(format, *args):
    l = []
    l.append(format % args)
    print(l[0])
i = 7
pi = 3.14159265359
#printf("hi there, i=%d, pi=%.2f", i, pi)
print("%10s"% ("Hola"))
l = []
length=15.0
l.append(("%"+str(length)+"s"% ("Hola")))
print(l[0])
