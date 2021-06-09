import matplotlib.pyplot as plt
import numpy as np; np.random.seed(1)

t = "Plot a pie chart  with different sized pies all in one figure"
X  = np.random.rand(12,4)*30
r = np.random.rand(12)*0.8+0.6

fig, axes= plt.subplots(3, 4)

for i, ax in enumerate(axes.flatten()):
    if i==4:
        break
    x = X[i,:]/np.sum(X[i,:])
    ax.pie(x, radius = 1.2, autopct="%.1f%%", pctdistance=0.9)
    ax.set_title(t.split()[i])
plt.savefig('example')
