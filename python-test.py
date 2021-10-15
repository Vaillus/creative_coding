import matplotlib.pyplot as plt
import numpy as np

cos = np.cos
pi = np.pi
x = np.arange(-np.pi, np.pi, 0.01)
"""sin = np.sin(x)
#cos = np.cos(x *2)
tes = np.clip(np.sin(x *2 - np.pi/2), 0, 1)
comp = np.clip(cos + tes, 0, 1)
compneg = comp.copy()

compneg[id:] = 0

comppos = (cos - tes).copy()
comppos[:id] = 0
tot = compneg + comppos"""
id = int(np.floor(len(x)/2))
left = cos(x*2)
left[id:] = 0
right = np.ones(len(left))
right[:id] = 0

plt.plot(x,np.sin(x*4/3- pi/6))
plt.show()

def get_bornes(n):
    """there are n+2 bornes

    Args:
        n ([type]): [description]

    Returns:
        [type]: [description]
    """
    bornes = []
    for i in range(n+2):
        bornes += [-pi + 2*pi*i/(n+1)]
    return bornes