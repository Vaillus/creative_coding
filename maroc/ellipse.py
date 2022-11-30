from __future__ import annotations
from math import atan2
import numpy as np
from typing import Tuple, List, Optional
from shapely.geometry.polygon import LinearRing
import matplotlib.pyplot as plt
#import cv2



class Arc():
    def __init__(
        self, 
        center: Tuple[float, float], 
        a: Optional[float]=None, 
        b: Optional[float]=None, 
        top_point: Optional[Tuple[float, float]]=None, 
        bottom_point: Optional[Tuple[float, float]]=None
    ):
        """A arc can be initialized in two ways:
        1. with the center and the two axis values, which are enough to 
           compute the equation of the ellipse.
        2. with the center and two points through which the ellipse 
           passes. The foci are computed from these axis values.
           
        a : horizontal axis value
        b : vertical axis value
        """
        # Either a and b or top_point and bottom_point must be specified
        if (a is None or b is None) and \
            (top_point is None or bottom_point is None):
            raise ValueError("Either a and b or top_point and bottom_point \
                must be provided")
        # If a and b are not specified, compute them from the two points
        if a is None:
            a, b = Arc.compute_ellipse_axis(center, top_point, bottom_point)
        self.center = center
        # check that a and b are not tuples
        assert type(a) is not tuple, "a is a tuple"
        self.a = a
        self.b = b
        self.tp = top_point
        self.bp = bottom_point

    @staticmethod
    def compute_ellipse_axis(
        center:Tuple[float], 
        p1:Tuple[float], 
        p2:Tuple[float]
    ) -> Tuple[float, float]:
        # print("compute a and b")
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
    
    def convert_point_rad(self, pt: Tuple[float]) -> float:
        """get a point on the ellipse and convert it to the radius from the center
        of the ellipse"""
        # get the angle of the point from the center of the ellipse
        angle = atan2((pt[1]-self.center[1])/self.b, (pt[0]-self.center[0])/self.a)
        return angle

    def convert_rad_point(self, angles:List[float]) -> List[Tuple[float]]:
        """convert a list of angles in radians to a list of points on the ellipse
        """
        x = []
        y = []
        for angle in angles:
            x.append(self.center[0] + self.a * np.cos(angle))
            y.append(self.center[1] + self.b * np.sin(angle))
        return list(zip(x, y))
    

    # === intersection function ========================================




    def intersect(self, other:Arc) -> Tuple[float, float]:
        """compute intersection between the current ellipse and the one
        provided as argument
        """
        # convert the ellipses into lists of points
        cur_poly = self.ellipse_polyline(angle=0.0)
        other_poly = other.ellipse_polyline(angle=0.0)
        # find the intersection points between the two ellipses
        x, y = Arc.intersections(cur_poly, other_poly)
        # find the point that is between the two limit points of ell2
        x, y = other._sel_good_point(x, y)
        return x, y

    def ellipse_polyline(
        self,
        angle:float=0.0,
        n:int=100
    ) -> np.ndarray[int, np.dtype[np.float64]]:
        t = np.linspace(0, 2*np.pi, n, endpoint=False)
        st = np.sin(t)
        ct = np.cos(t)
        angle = np.deg2rad(angle)
        sa = np.sin(angle)
        ca = np.cos(angle)
        result = np.empty((n, 2))
        result[:, 0] = self.center[0] + self.a * ca * ct - self.b * sa * st
        result[:, 1] = self.center[1] + self.a * sa * ct + self.b * ca * st
        return result

    def _sel_good_point(
        self, 
        x:List[float], 
        y:List[float], 
    ) -> Tuple[float]:
        """find the point that is between the two limit points of ell2
        """
        xmin = min(self.tp[0], self.bp[0])
        xmax = max(self.tp[0], self.bp[0])
        ymin = min(self.tp[1], self.bp[1])
        ymax = max(self.tp[1], self.bp[1])
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

    @staticmethod
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



    # === rendering ====================================================



    def render(
        self, 
        img: np.ndarray[int, np.dtype[np.int64]], 
        color: Tuple[int]=(0,0,0), 
        bold: bool=False
    ) -> None:
        xmax = max(self.tp[0],self.bp[0])
        xmin = min(self.tp[0],self.bp[0])
        ymax = max(self.tp[1],self.bp[1])
        ymin = min(self.tp[1],self.bp[1])
        for x in range(int(-self.a), int(self.a)):
            if (x+self.center[0] <= xmax) and (x+self.center[0]>=xmin):
                yp = self.b * np.sqrt(1 - (x/self.a)**2)
                ym = - yp
                if (yp+self.center[1] <= ymax) and (yp+self.center[1]>=ymin):
                    img[int(x+self.center[0]), int(yp+self.center[1])] = color
                    # draw all adjacent pixels
                    self._draw_bold_circle(img, x, yp, color, bold=bold)
                if (ym+self.center[1] <= ymax) and (ym+self.center[1]>=ymin):
                    img[int(x+self.center[0]),int(ym+self.center[1])] = color
                    self._draw_bold_circle(img, x, ym, color, bold=bold)

    def _draw_bold_circle(
        self, 
        img:np.ndarray[int, np.dtype[np.int64]], 
        x: int, 
        y: int,
        color: Tuple[int], 
        bold:bool
    ) -> None:
        if bold:
            cv2.circle(
                img, 
                (int(x+self.center[0]), int(y+self.center[1])), 
                2, 
                color, 
                -1
            )
    
    def get_points(self, n_points:int) -> List[Tuple[float]]:
        """return n_point points on the ellipse between top and bottom points
        """
        # # ! absolutely not sure that it is going to work
        # start_x = min(self.tp[0], self.bp[0])
        # end_x = max(self.tp[0], self.bp[0])
        # points = []
        # for i in range(n_points):
        #     x = start_x + (i * (end_x - start_x) / (n_points - 1))
        #     # y is the value of the ellipse at x, given the center is self.center
        #     y = self.b * np.sqrt(1 - ((x - self.center[0])/self.a)**2) + self.center[1]
        #     points.append((x, y))

        # compute the angle of the top and bottom points
        angle_tp = self.convert_point_rad(self.tp)
        angle_bp = self.convert_point_rad(self.bp)
        # plot lines goin from center and going in the direction of the top and bottom points
        # use the ang_tp and ang_bp to compute the angle of the line

        # get the n_points intermediate angles between the two angles, using the
        # sortest path on the circle
        # generat angle from 0 to 2pi
        #angles = np.linspace(0, 2*np.pi, 100)
        angles = np.linspace(angle_tp, angle_bp, n_points)
        #angle = np.linspace(angle_tp, angle_bp, n_points)
        # convert the angles to points
        points = self.convert_rad_point(angles)
        return points

    