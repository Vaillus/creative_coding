import processing.core.*; 
import processing.data.*; 
import processing.event.*; 
import processing.opengl.*; 

import java.util.HashMap; 
import java.util.ArrayList; 
import java.io.File; 
import java.io.BufferedReader; 
import java.io.PrintWriter; 
import java.io.InputStream; 
import java.io.OutputStream; 
import java.io.IOException; 

public class creative_coding extends PApplet {

float r = 100;



public void setup()
{
  
}

int numFrames = 60;

public void draw()
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


public void step()
{
  background(255);
  
  float t = 1.0f*frameCount/numFrames;
  
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


public float periodicFunction(float p)
{
  return map(sin(TWO_PI*p),-1,1,2,8);
}

public float offset(float x,float y)
{
  return 0.01f*dist(x,y,width/2,height/2);
}
  public void settings() {  size(500,500); }
  static public void main(String[] passedArgs) {
    String[] appletArgs = new String[] { "creative_coding" };
    if (passedArgs != null) {
      PApplet.main(concat(appletArgs, passedArgs));
    } else {
      PApplet.main(appletArgs);
    }
  }
}
