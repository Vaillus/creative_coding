from math import atan2
import numpy as np

def draw_arc(center, a,b, p1, p2, img):
    xmax = max(p1[0],p2[0])
    xmin = min(p1[0],p2[0])
    ymax = max(p1[1],p2[1])
    ymin = min(p1[1],p2[1])
    for x in range(int(-a), int(a)):
        if (x+center[0] <= xmax) and (x+center[0]>=xmin):
            yp = b * np.sqrt(1 - (x/a)**2)
            ym = - yp
            if (yp+center[1] <= ymax) and (yp+center[1]>=ymin):
                #point(x+center[0], yp+center[1])
                # draw point with cv2
                #cv2.circle(img, (x+center[0], yp+center[1]), 1, (0,0,255), -1)
                img[int(x+center[0]), int(yp+center[1])] = (0,0,255)
            if (ym+center[1] <= ymax) and (ym+center[1]>=ymin):
                img[int(x+center[0]),int(ym+center[1])] = (0,0,255)
                #cv2.circle(img, (x+center[0], yp+center[1]), 1, (0,0,255), -1)
                #point(x+center[0],ym+center[1])

class Ellipsex():
    def __init__(self, center, a, b, top_point=None, bottom_point=None):
        self.center = center
        self.a = a
        self.b = b
        self.tp = top_point
        self.bp = bottom_point
        #self.tang = self.convert_point_rad(self.tp)
        #self.bang = self.convert_point_rad(self.bp)

    @staticmethod
    def compute_ellipse_from_three(center, p1, p2):
        #print("compute a and b")
        Ax = (p1[0]- center[0]) ** 2
        Ay = (p1[1]- center[1]) ** 2
        Bx = (p2[0]- center[0]) ** 2
        By = (p2[1]- center[1]) ** 2
        #eps = 1
        num = Ay - By
        den = Bx - Ax
        # handle exceptions
        o = float(num)/float(den)
        #print("Ax:"+str(Ax))
        #print("Ay:"+str(Ay))
        #print("Bx:"+str(Bx))
        #print("By:"+str(By))
        #print("o:"+str(o))
        #print(Ax - Ay/o)
        a = np.sqrt(float(Ax) + float(Ay)/o)
        #print("s")
        b = np.sqrt(float(Ax)*o + float(Ay))
        #print("a: "+str(a))
        #print("b: "+str(b))
        
        return a, b

    def display(self, img):
        draw_arc(self.center, self.a, self.b, self.tp, self.bp, img)
    
    def convert_point_rad(self, pt):
        res = atan2(pt[1] - self.center[1], pt[0] - self.center[0])
        #print("relative x axis : "+str(pt[0] - self.center[0]))
        #print("relative y axis : "+str(pt[1] - self.center[1]))
        #print("angle: "+str(res))
        return res
    
def intersect(el1:Ellipsex, el2:Ellipsex):
    # compute the intersection points between the two ellipses
    x = el1.center[0]
    y = el1.center[1]
    a = el1.a
    b = el1.b
    x2 = el2.center[0]
    y2 = el2.center[1]
    a2 = el2.a
    b2 = el2.b


class Arcx(Ellipsex):
    def __init__(self, center, top_point=None, bottom_point=None, start_angle=None, end_angle=None):
        a, b = Ellipsex.compute_ellipse_from_three(center, top_point, bottom_point)
        super().__init__(center, a, b, top_point, bottom_point)
        self.start_angle = start_angle
        self.end_angle = end_angle

    def display(self, img):
        draw_arc(self.center, self.a, self.b, self.tp, self.bp, img)