from collections import defaultdict
d = dict.fromkeys("hola1,hola2".split(","))
a = dict.fromkeys("hola1,hola2".split(","))
f = dict.fromkeys("hola1,hola2".split(","))
g = dict.fromkeys("hola1,hola2".split(","))
def init(d):
    for key in d:
        d[key] = list()
    for k1,k2,k3,k4 in zip(d,a,f,g): 
        d[k1] = list()
        a[k2] = list()
        f[k3] = list()
        g[k4] = list()

def add(d : dict, key, value):
    d[key].append(value)

init(d)
print(d)
add(d,"hola1",1)
print(d)
add(d,"hola1",2)
add(d,"hola1",3)
print(d)
print(g)