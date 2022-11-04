from math import atan2
import numpy as np
from typing import Tuple, List
from shapely.geometry.polygon import LinearRing
import matplotlib.pyplot as plt

def draw_arc(center, a,b, p1, p2, img, color=(0,0,0)):
    xmax = max(p1[0],p2[0])
    xmin = min(p1[0],p2[0])
    ymax = max(p1[1],p2[1])
    ymin = min(p1[1],p2[1])
    for x in range(int(-a), int(a)):
        if (x+center[0] <= xmax) and (x+center[0]>=xmin):
            yp = b * np.sqrt(1 - (x/a)**2)
            ym = - yp
            if (yp+center[1] <= ymax) and (yp+center[1]>=ymin):
                img[int(x+center[0]), int(yp+center[1])] = color
            if (ym+center[1] <= ymax) and (ym+center[1]>=ymin):
                img[int(x+center[0]),int(ym+center[1])] = color

class Oval():
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
        a = np.sqrt(float(Ax) + float(Ay)/o)
        b = np.sqrt(float(Ax)*o + float(Ay))
        
        return a, b

    def display(self, img):
        draw_arc(self.center, self.a, self.b, self.tp, self.bp, img)
    
    def convert_point_rad(self, pt):
        res = atan2(pt[1] - self.center[1], pt[0] - self.center[0])
        return res
    
    def intersect(self, other) -> Tuple[float]:
        # convert the ellipses into a list of points
        a, b = ellipse_polyline(
            [(self.center[0], self.center[1], self.a, self.b, 0), 
            (other.center[0], other.center[1], other.a, other.b, 0)]
        )
        # find the intersection points between the two ellipses
        x, y = intersections(a, b)
        # find the point that is between the two limit points of ell2
        x, y = sel_good_point(x, y, other)
        return x, y

    def render(self, img, color=(0,0,0)):
        draw_arc(self.center, self.a, self.b, self.tp, self.bp, img, color)




class Arc(Oval):
    def __init__(self, center, top_point=None, bottom_point=None, start_angle=None, end_angle=None):
        a, b = Oval.compute_ellipse_from_three(center, top_point, bottom_point)
        super().__init__(center, a, b, top_point, bottom_point)
        self.start_angle = start_angle
        self.end_angle = end_angle

def sel_good_point(x:List[float], y:List[float], ell2) -> Tuple[float]:
    """find the point that is between the two limit points of ell2

    Args:
        x (List[float]): 
        y (List[float]): 
        ell2 (el.Oval):

    Returns:
        Tuple[float]: 
    """
    xmin = min(ell2.tp[0], ell2.bp[0])
    xmax = max(ell2.tp[0], ell2.bp[0])
    ymin = min(ell2.tp[1], ell2.bp[1])
    ymax = max(ell2.tp[1], ell2.bp[1])
    for i in range(len(x)):
        xi = x[i]
        yi = y[i]
        if xi <= xmax and xi >= xmin and yi <= ymax and yi >= ymin:
            x = xi
            y = yi
            break
    # assert type of x is float
    if type(x) != float:
        # select value closest to x between xmin and xmax
        xsel = min(x, key=lambda x:abs(x-xmin))
        # ysel equals the corresponding y value
        ysel = y[x.index(xsel)]
        x=xsel
        y=ysel
    assert type(x) is float, "x is not a float"
    return x, y




def ellipse_polyline(ellipses, n=100):
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    st = np.sin(t)
    ct = np.cos(t)
    result = []
    for x0, y0, a, b, angle in ellipses:
        angle = np.deg2rad(angle)
        sa = np.sin(angle)
        ca = np.cos(angle)
        p = np.empty((n, 2))
        p[:, 0] = x0 + a * ca * ct - b * sa * st
        p[:, 1] = y0 + a * sa * ct + b * ca * st
        result.append(p)
    return result

def intersections(a, b):
    ea = LinearRing(a)
    eb = LinearRing(b)
    mp = ea.intersection(eb)
    try:
        x = [p.x for p in mp]
        y = [p.y for p in mp]
    except:
        print("no intersection")
        plt.plot(a[:,0], a[:,1])
        plt.plot(b[:,0], b[:,1])
        plt.show()
        
    return x, y