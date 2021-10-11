float r = 100;



void setup()
{
  size(500,500);
}

int numFrames = 60;

void draw()
{
  step();
  
  if(frameCount<=numFrames)
  {
    saveFrame("fr###.gif");
  }
  if(frameCount==numFrames)
  {
    println("All frames have been saved");
  }
}


void step()
{
  background(255);
  
  float t = 1.0*frameCount/numFrames;
  
  int m = 20;
  
  stroke(0);
  
  for(int i=0;i<m;i++)
  {
    for(int j=0;j<m;j++)
    {
      float x = map(i,0,m-1,0,width);
      float y = map(j,0,m-1,0,height);
      
      float size = periodicFunction(t-offset(x,y));
      strokeWeight(size);      
      point(x,y);
    }
  }
}


float periodicFunction(float p)
{
  return map(sin(TWO_PI*p),-1,1,2,8);
}

float offset(float x,float y)
{
  return 0.01*dist(x,y,width/2,height/2);
}
