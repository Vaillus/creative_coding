import numpy as np
import cv2
from typing import Tuple, List, Optional
from math import atan2

class Goutte:
    def __init__(self, height, angle):
        self.hei = height
        self.ang = angle
        self.rad = self._circle_radius()
        self.wid = 2 * self.rad
        self.tri_hei = self._triangle_height()
        self.tri_wid = self._triangle_width()
        self.d_tri_cir = self._dist_tri_center_circle()
        assert self.hei == self.rad + self.d_tri_cir + self.tri_hei, "error in calculus"
        self.top_pt = None
        self.bot_pt = None
        self.l_pt = None
        self.r_pt = None
        self.center = None
        self._init_points()

    def _circle_radius(self):
        r = self.hei * (np.sin(self.ang/2))/(1 + np.sin(self.ang/2))
        return r
    
    def _triangle_height(self):
        h = self.hei * (1.0 - np.sin(self.ang / 2.0))
        return h

    def _triangle_width(self):
        w = self.rad * np.cos(self.ang / 2.0)
        return w

    def _dist_tri_center_circle(self):
        d = self.rad * np.sin(self.ang / 2.0)
        return d

    def _init_points(self):
        self.bot_pt = (self.wid / 2.0, self.rad)
        self.top_pt = (self.wid / 2.0, self.hei)
        self.l_pt = (self.wid / 2.0 - self.tri_wid, self.rad + self.d_tri_cir)
        self.r_pt = (self.wid / 2.0 + self.tri_wid, self.rad + self.d_tri_cir)
    
    def render(self, img: np.ndarray[int, np.dtype[np.int32]]):
        img = cv2.line(img, tup2int(self.top_pt), tup2int(self.l_pt), (0, 0, 0))
        img = cv2.line(img, tup2int(self.top_pt), tup2int(self.r_pt), (0,0,0))
        lpt_ang = point2rad(self.bot_pt, self.l_pt)
        rpt_ang = point2rad(self.bot_pt, self.r_pt)
        ang_vec = gen_largest_arc(lpt_ang, rpt_ang)
        #ang_vec = [lpt_ang, rpt_ang]
        pts = rad2point(self.bot_pt, self.rad, ang_vec)
        for pt in pts:
            img[int(pt[0]), int(pt[1]), :] = 0
        # img = cv2.circle(
        #     img, 
        #     tup2int(self.bot_pt), 
        #     int(self.rad), 
        #     (0, 0, 0), 
        #     1
        # )

def gen_largest_arc(ang1, ang2):
    n_points = 300
    ang_sub = min(ang1, ang2)
    ang1 -= ang_sub
    ang2 -= ang_sub
    not_zero_ang = max(ang1, ang2)
    clockwise = not_zero_ang <= np.pi 
    if clockwise:
        angs = np.linspace(not_zero_ang, 2*np.pi, n_points)
    else:
        angs = np.linspace(0.0, not_zero_ang, n_points)
    angs += ang_sub
    return angs

def tup2int(tup):
    n_tup = tuple([int(x) for x in tup])
    return n_tup

def point2rad(center: Tuple[float, float], pt: Tuple[float, float]) -> float:
        """get a point on the circle and convert it to the radius from the center
        of the ellipse"""
        # get the angle of the point from the center of the ellipse
        angle = atan2(pt[0]-center[0], pt[1]-center[1])
        return angle

def rad2point(
    center: Tuple[float, float], 
    rad: float,
    angles:List[float]
) -> List[Tuple[float]]:
    """convert a list of angles in radians to a list of points on the ellipse
    """
    x = []
    y = []
    for angle in angles:
        x.append(center[0] + rad * np.cos(angle))
        y.append(center[1] + rad * np.sin(angle))
    return list(zip(x, y))

if __name__ == "__main__":
    ang = np.deg2rad(45)
    goutte = Goutte(300, ang)
    img = np.ones((400, 400, 3), dtype = "uint8") * 255
    goutte.render(img)
    img = img[::-1,:,:]
    while True:
        cv2.imshow("output", img)
        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            break

    

