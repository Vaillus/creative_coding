float r = 100;

void setup()
{
  size(400,400);
}

int numFrames = 100;

void draw()
{
  background(255);
  
  float t = 1.0*frameCount/numFrames;
  
  float r = 100;
  float x = width/4 + r*cos(TWO_PI*t);
  float y = height/2 + r*sin(TWO_PI*t);
  
  stroke(0);

  strokeWeight(9.0); // size of black dot
  
  point(x,y);
  
  if(frameCount<=numFrames)
  {
    saveFrame("gifs/fr###.gif");
  }
  if(frameCount==numFrames)
  {
    println("All frames have been saved");
  }
}
