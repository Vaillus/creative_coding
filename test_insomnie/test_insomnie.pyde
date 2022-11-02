from typing import Tuple
import math
def setup():
    size(400,800)

numFrames = 100
blu = [13, 0, 11]
re = [31, 20, 82]

def gradient(f, c1, c2):
    ret = []
    for i in range(3):
        comp1 = c1[i]
        comp2 = c2[i]
        diff = abs(comp1 - comp2)
        sc = f * diff
        val = min(comp1, comp2) + sc
        ret += [val]
    return ret

def scale_arange(arange):
    maxi = len(arange) - 1
    #print(arange)
    return [map(i, 0, maxi, 0, TWO_PI) for i in arange]

def sin_arange(arange, offset):
    # take as input values between -1 and 1 and an offset from
    return [map(sin(i + offset),-1,1, 0.4,1) for i in arange]

def scale_i(i, maxi, mini):
    return (i- mini)/(maxi - mini)

def draw():
    background(255)
    t = 1.0*frameCount/numFrames
    n_franges = 30
    arange = [i for i in range(n_franges)]
    #print(arange)
    arange = scale_arange(arange)
    arange = sin_arange(arange,t*TWO_PI)
    sum = 0
    for i in arange:
        sum += i
    arange = [i / sum for i in arange]
    sum2=0
    maxi = max(arange)
    mini = min(arange)
    for i in arange:
        sum2 += i
    #print(sum2)
    cumul = 0
    #print(arange)
    for i in arange:
        #print(i)
        h = i * height
        #print((1- (i/maxi)))
        i=scale_i(i, maxi, mini)
        c = gradient(i, blu, re)#color(i*150, 0, (1- i)*100)
        ##print(c)
        fill(color(*c))
        rect(0, cumul, width, h)
        cumul += h
    if(frameCount<=numFrames):
        saveFrame("gif/fr###.gif")



def compute_angle(x:tuple, y:tuple):
    x1, y1 = x
    x2, y2 = y
    return atan2(y2-y1, x2-x1)
