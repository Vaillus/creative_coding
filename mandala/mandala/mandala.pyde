from Test import *

numFrames = 500

def setup():
    size(400,400)
    test()

def draw():
    t = 1.0*frameCount/numFrames
    background(255)
    #point(10,10)
    test_func(t)

    #if(frameCount<=numFrames):
    #    saveFrame("gif/fr###.gif")
    #if frameCount == numFrames:
    #    print("export done.")
