import numpy as np
import cv2
import math
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt
from PIL import Image

import maroc.toolkit.toolkit as tk
from maroc.toolkit.line import Line
from maroc.lampe.arc import Arc
import maroc.goutte.texture as tex

class Goutte:
    def __init__(
        self, 
        height: float, 
        angle: float, 
        center: Tuple[float, float], 
        width: float = 1.0, 
        debug: bool = False
    ):
        self.hei = height # hauteur de la goutte
        self.ang = angle # angle du cône (rad)
        self.alpha = self.ang / 2.0
        self.y1 = self.hei * (1.0 - np.sin(self.alpha))
        self.x1 = self.y1 * np.tan(self.alpha)
        self.k = float(self.hei * (1.0-np.sin(self.alpha)) / (np.cos(self.alpha)**2))
        self.rad = self.hei - self.k
        # self.rad = self._circle_radius() # rayon du cercle à la base
        self.cir_center = (0.0, self.rad)
        # self.wid = 2.0 * self.rad # largeur de la goutte
        # self.tri_hei = self._triangle_height() # hauteur du triangle au sommet
        # self.tri_wid = self._triangle_width() # largeur de la base du triangle
        # self.d_tri_cir = self._dist_tri_center_circle() # distance entre 
        # le centre du cercle et la base du triangle
        # assert self.hei == self.rad + self.d_tri_cir + self.tri_hei,\
            #   "error in calculus"
        self.bot_pt, self.top_pt, self.l_pt, self.r_pt = self._init_points()
        # moving the figure such that its center is at the given center
        # compute the offset of the center of the figure from the given 
        # center
        init_center = (
            (self.r_pt[0] + self.l_pt[0]) / 2.0, 
            (self.top_pt[1] + self.bot_pt[1]) / 2.0
        )
        diff = np.subtract(center, init_center)
        # offset all points
        self.bot_pt = np.add(self.bot_pt, diff)
        self.top_pt = np.add(self.top_pt, diff)
        self.l_pt = np.add(self.l_pt, diff)
        self.r_pt = np.add(self.r_pt, diff)
        self.cir_center = np.add(self.cir_center, diff)
        self.center = center
        self.width = width
        self.arc = self.init_arc()
        self.right_line = Line(self.r_pt, self.top_pt)
        self.left_line = Line(self.l_pt, self.top_pt)
        # determines wether debug points are displayed.
        self.debug = debug


    # === initialization ===============================================



    def _circle_radius(self):
        """ Compute the radius of the circle at the bottom of the drop 
        from the height of the figure and the aperture of the top angle.
        """
        r = self.hei * np.sin(self.alpha) / (np.cos(self.alpha)**2)
        r = float(r)
        return r
    
    def _triangle_height(self):
        """ Compute the height of the triangle at the top of the drop 
        from the height of the figure and the aperture of the top angle.
        """
        h = float(self.hei * (1.0 - np.sin(self.ang / 2.0)))
        return h

    def _triangle_width(self):
        """ Compute the width of the triangle at the top of the drop 
        from the radius of the circle at the bottom of the drop and 
        the aperture of the top angle.
        """
        w = float(self.rad * np.cos(self.ang / 2.0))
        return w

    def _dist_tri_center_circle(self):
        """ Compute the distance between the center of the circle at the 
        bottom of the drop and the bottom of the triangle at the top of 
        the drop.
        """
        d = float(self.rad * np.sin(self.ang / 2.0))
        return d

    def _init_points(self):
        bot_pt = (0.0, 0.0)
        top_pt = (0.0, self.hei)
        l_pt = (-self.x1, self.hei - self.y1)
        r_pt = (self.x1, self.hei - self.y1)
        return bot_pt, top_pt, l_pt, r_pt
    
    def init_arc(self) -> Arc:
        lpt_ang = tk.point2rad(self.cir_center, self.l_pt)
        rpt_ang = tk.point2rad(self.cir_center, self.r_pt)
        arc = Arc(
            center=self.cir_center, 
            a=self.rad, 
            b=self.rad, 
            ang_end=lpt_ang, 
            ang_start=rpt_ang
        )
        return arc
    
    @staticmethod
    def scaled_goutte(ref_goutte: 'Goutte', scale: float):
        """initialize a new Goutte inside another one such that they are 
        correctly spaced from one another. (could also be outside)"""
        # Step 1: Compute the new radius and the new bottom point
        # circ_center = np.add(ref_goutte.bot_pt, (0.0, ref_goutte.rad))
        circ_center = ref_goutte.cir_center
        new_rad = ref_goutte.rad * scale
        new_bot_pt = np.subtract(circ_center, (0.0, new_rad))
        # Step 2: Compute the new left and right points. 
        # They are aligned with the old ones on the axes from the center 
        # of the circle.
        lpt_ang = tk.point2rad(circ_center, ref_goutte.l_pt)
        rpt_ang = tk.point2rad(circ_center, ref_goutte.r_pt)
        new_l_pt = tk.rad2point(circ_center, new_rad, lpt_ang)
        new_r_pt = tk.rad2point(circ_center, new_rad, rpt_ang)
        # Step 3: Compute the position of the top point as the 
        # intersections of the tangents on the new circle that pass by
        # the new left and right points.
        dxl = -(new_l_pt[1] - circ_center[1])
        dyl = new_l_pt[0] - circ_center[0]
        dxr = -(new_r_pt[1] - circ_center[1])
        dyr = new_r_pt[0] - circ_center[0]
        # Coefficients for the tangent at new_l_pt
        A1 = dyl
        B1 = -dxl
        C1 = dyl * new_l_pt[0] - dxl * new_l_pt[1]
        # Coefficients for the tangent at p2
        A2 = dyr
        B2 = -dxr
        C2 = dyr * new_r_pt[0] - dxr * new_r_pt[1]
        # Solve the system of equations
        determinant = A1 * B2 - A2 * B1
        if determinant == 0:
            print("The tangents are parallel and do not intersect.")
        else:
            x3 = (C1 * B2 - C2 * B1) / determinant
            y3 = (A1 * C2 - A2 * C1) / determinant
            new_top_pt = (x3, y3)
            # tk.render_debug_point(img, new_top_pt)
        # Step 4: Compute the arguments of the new Goutte from the new 
        # points
        new_hei = new_top_pt[1] - new_bot_pt[1]
        new_wid = new_r_pt[0] - new_l_pt[0]
        tri_hei = new_top_pt[1] - new_l_pt[1]
        new_ang = 2 * math.atan2(new_wid/2, tri_hei)
        # Goutte.render_angle_test(img, new_top_pt, new_ang)
        new_center = tk.interpolate_pts(new_bot_pt, new_top_pt, 0.5)
        # Step 5: Return the new Goutte
        return Goutte(new_hei, new_ang, new_center)
    
    # @staticmethod
    # def render_angle_test(img: np.ndarray[int, np.dtype[np.int32]], pt: Tuple[float, float], ang: float):
    #     """afficher les deux demi-droites issues de pt et formant un angle ang vers le bas."""
    #     tk.render_debug_point(img, pt)
    #     pt1 = tk.rad2point(pt, 100, - np.pi/2 - ang/2)
    #     pt2 = tk.rad2point(pt, 100, - np.pi/2 + ang/2)
    #     tk.render_debug_line(img, pt, pt1, 'green')
    #     tk.render_debug_line(img, pt, pt2, 'green')
        


    # === rendering ====================================================



    def render(self, img: np.ndarray[int, np.dtype[np.int32]]):
        # render the two diagonal top lines of the drop
        img = self.right_line.render(img, (0,0,0), self.width)
        img = self.left_line.render(img, (0,0,0), self.width)
        # render the bottom curve of the drop
        self.arc.render(img, (0,0,0), width=self.width)
        # ang_vec = Goutte.gen_largest_arc(lpt_ang, rpt_ang)
        # ang_vec = [lpt_ang, rpt_ang]
        # pts = Goutte.rad2point(cir_center, self.rad, ang_vec)
        # for pt in pts:
        #     tk.set_pixel(img, *pt, (0, 0, 0))
        # plot the four main points in red
        if self.debug:  
            tk.render_debug_point(img, self.bot_pt)
            tk.render_debug_point(img, self.top_pt)
            tk.render_debug_point(img, self.l_pt)
            tk.render_debug_point(img, self.r_pt)
            tk.render_debug_point(img, cir_center)


if __name__ == "__main__":
    siz = 401
    ang = float(np.deg2rad(37.0))
    hei = 3/4*siz

    goutte = Goutte(hei, ang, (siz/2, siz/2), width = 1)
    goutte2 = Goutte.scaled_goutte(goutte, 0.89)
    goutte.width = 3
    goutte2.width = 2
    img = np.ones((siz, siz, 3), dtype = "uint8") * 255
    goutte.render(img)
    goutte2.render(img)
    mask = tk.flood_fill_mask(img, (int(siz/2), int(siz/2)))
    img2 = np.zeros_like(img) + 255
    tex = tex.HexTexture(siz, siz, 2/3)
    imgs = []

    t = 0
    T = 53
    while t < T:
        t+=1
        tex.add_to_pos(1)
        img = np.ones((siz, siz, 3), dtype = "uint8") * 255
        img2 = np.zeros_like(img) + 255
        goutte.render(img)
        goutte2.render(img)
        img2 = tex.render(img2)
        # use the mask to fill img with img2 where mask is True
        img[mask] = img2[mask]
        # tk.render_debug_point(img, goutte.top_pt, small=True, color='red')
        # tk.render_debug_point(img, goutte.l_pt, small=True, color='red')

        tk.render(img)
        tk.add_frame(imgs, img)
        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            break
    tk.save_gif(imgs, 'test.gif')
    

