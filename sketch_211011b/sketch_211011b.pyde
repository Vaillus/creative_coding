

def setup():
    size(300, 300)
    background(255)
    smooth()
    
    noFill()
    polygon(3, 50, 75, 100, 100, -PI / 2.0)   # -90 degrees
    polygon(4, 170, 75, 50, 125, -PI / 4.0)   # -45 degrees
    
    fill(255, 204, 255)
    stroke(128, 0, 128)
    polygon(5, 50, 200, 75, 50, -PI / 2.0)    # -90 degrees
    
    noFill()
    stroke(0)
    polygon(6, 170, 200, 50, 100, 0)

    stroke(128)
    # draw enclosing ellipses to make sure we did it right
    ellipse(50, 75, 100, 100)
    ellipse(170, 75, 50, 125)
    ellipse(50, 200, 75, 50)
    ellipse(170, 200, 50, 100)

def polygon(n, cx, cy, w, h, startAngle):
    angle = TWO_PI/ n
    
    # The "radius" is one half the total width and height
    w = w / 2.0
    h = h / 2.0
    
    beginShape()
    for i in xrange(n):
        vertex(cx + w * cos(startAngle + angle * i),
          cy + h * sin(startAngle + angle * i))
    endShape(CLOSE)
