from math import atan2
import numpy as np
from typing import Tuple, List, Optional
from shapely.geometry.polygon import LinearRing
import matplotlib.pyplot as plt
import cv2
from __future__ import annotations

def draw_arc(
    center, 
    a: float,
    b: float, 
    p1: Tuple[float], 
    p2: Tuple[float], 
    img: np.ndarray[int, np.dtype[np.float64]], 
    color: Tuple[int]=(0,0,0), 
    bold:bool=False
) -> None:
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
                # draw all adjacent pixels
                if bold:
                    cv2.circle(
                        img, 
                        (int(x+center[0]), int(yp+center[1])), 
                        2, 
                        color, 
                        -1
                    )
            if (ym+center[1] <= ymax) and (ym+center[1]>=ymin):
                img[int(x+center[0]),int(ym+center[1])] = color
                if bold:
                    # draw all adjacent pixels
                    cv2.circle(
                        img, 
                        (int(x+center[0]), int(ym+center[1])), 
                        2, 
                        color, 
                        -1
                    )

class Oval():
    def __init__(
        self, 
        center: Tuple[float], 
        a: float, 
        b: float, 
        top_point: Optional[Tuple[float]]=None, 
        bottom_point: Optional[Tuple[float]]=None
    ):
        self.center = center
        self.a = a
        self.b = b
        self.tp = top_point
        self.bp = bottom_point

    @staticmethod
    def compute_ellipse_axis(
        center:Tuple[float], 
        p1:Tuple[float], 
        p2:Tuple[float]
    ) -> Tuple[float]:
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

    def display(
        self, 
        img: np.ndarray[int, np.dtype[np.int64]], 
        bold:bool=False
    ) -> None:
        draw_arc(self.center, self.a, self.b, self.tp, self.bp, img, bold=bold)
    
    def convert_point_rad(self, pt: Tuple[float]) -> float:
        res = atan2(pt[1] - self.center[1], pt[0] - self.center[0])
        return res
    
    def intersect(self, other:Oval) -> Tuple[float]:
        # convert the ellipses into a list of points
        a, b = ellipse_polyline(
            [(self.center[0], self.center[1], self.a, self.b, 0.0), 
            (other.center[0], other.center[1], other.a, other.b, 0.0)]
        )
        # find the intersection points between the two ellipses
        x, y = intersections(a, b)
        # find the point that is between the two limit points of ell2
        x, y = sel_good_point(x, y, other)
        return x, y

    def render(
        self, 
        img: np.ndarray[int, np.dtype[np.int64]], 
        color: Tuple[int]=(0,0,0), 
        bold: bool=False
    ) -> None:
        draw_arc(
            self.center, self.a, self.b, self.tp, self.bp, img, color, bold)




class Arc(Oval):
    def __init__(
        self, 
        center:Tuple[float], 
        top_point:Optional[Tuple[float]]=None, 
        bottom_point:Optional[Tuple[float]]=None, 
        start_angle:Optional[float]=None, 
        end_angle:Optional[float]=None
    ):
        a, b = Oval.compute_ellipse_axis(center, top_point, bottom_point)
        super().__init__(center, a, b, top_point, bottom_point)
        self.start_angle = start_angle
        self.end_angle = end_angle

def sel_good_point(x:List[float], y:List[float], ell2: Oval) -> Tuple[float]:
    """find the point that is between the two limit points of ell2
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

def ellipse_polyline(
    ellipses: List[Tuple[int]], 
    n:int=100
) -> List[np.ndarray[int, np.dtype[np.float64]]]:
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

def intersections(
    a: np.ndarray[int, np.dtype[np.float64]], 
    b: np.ndarray[int, np.dtype[np.float64]]
) -> Tuple[List[float]]:
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