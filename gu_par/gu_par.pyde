import centroides as ctd

numFrames = 500
w = 400
h = 400
til = ctd.Tiling(10, 10, w, h)

def setup():
    size(w,h)


def draw():
    t = 1.0*frameCount/numFrames
    background(255)
    #point(10,10)
    
    for c in til.centroids():
        print([c.neighbours.values()])
        c.move()
        point(c.x, c.y)
    print("a")

    #test_func(t)

    #if(frameCount<=numFrames):
    #    saveFrame("gif/fr###.gif")
    #if frameCount == numFrames:
    #    print("export done.")
