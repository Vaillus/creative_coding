def setup():
  size(500,500)

numFrames = 60;

def periodicFunction(p, s):
    val = sin(TWO_PI*p)
    if abs(val) > 0.2:
        sign = (val < 0.0) * -1.0 + (val > 0.0) * 1.0
        val = sign * 0.2 + val * 0.1
    return map(val,-1, 1,20,100)

def offset(s):
    return 0.01*s

def square_color(s):
    if s % 2 == 0:
        return color(0,0,0)
    else:
        return color(255,255,255)

def draw():
  background(0)
  t = 1.0*frameCount/numFrames
  ns =1
  squaresz = 50
  black = color(0,0,0)
  fill(255)
  strokeWeight(1)
  #square(width/2 - squaresz /2, height/2 - squaresz /2, squaresz)
  stroke(0)
  for s in range(ns):
    #fill(square_color(s))
    squaresz = periodicFunction(t,s)
    square(width/2 - squaresz /2, height/2 - squaresz /2, squaresz)
