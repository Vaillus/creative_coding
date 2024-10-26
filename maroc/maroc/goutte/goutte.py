import numpy as np
import cv2
import math
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt

import maroc.toolkit.toolkit as tk

class Goutte:
    def __init__(self, height, angle, center, debug=False):
        self.hei = height # hauteur de la goutte
        self.ang = angle # angle du cône (rad)
        self.rad = self._circle_radius() # rayon du cercle à la base
        self.wid = 2 * self.rad # largeur de la goutte
        self.tri_hei = self._triangle_height() # hauteur du triangle au sommet
        self.tri_wid = self._triangle_width() # largeur de la base du triangle
        self.d_tri_cir = self._dist_tri_center_circle() # distance entre 
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
        self.center = center
        self.debug = debug


    # === initialization ===============================================



    def _circle_radius(self):
        """ Compute the radius of the circle at the bottom of the drop 
        from the height of the figure and the aperture of the top angle.
        """
        r = float(self.hei * (np.sin(self.ang/2))/(1 + np.sin(self.ang/2)))
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
        bot_pt = (self.wid / 2.0, 0)
        top_pt = (self.wid / 2.0, self.hei)
        l_pt = (self.wid / 2.0 - self.tri_wid, self.rad + self.d_tri_cir)
        r_pt = (self.wid / 2.0 + self.tri_wid, self.rad + self.d_tri_cir)
        return bot_pt, top_pt, l_pt, r_pt
    
    @staticmethod
    def scaled_goutte(img, ref_goutte: 'Goutte', scale: float):
        """initialize a new Goutte inside another one such that they are 
        correctly spaced from one another. (could also be outside)"""
        # Step 1: Compute the new radius and the new bottom point
        circ_center = np.add(ref_goutte.bot_pt, (0, ref_goutte.rad))
        new_rad = ref_goutte.rad * scale
        new_bot_pt = np.subtract(circ_center, (0, new_rad))
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
        img = cv2.line(
            img, 
            tk.tup_float2int(self.top_pt), 
            tk.tup_float2int(self.l_pt), 
            (0,0,0)
        )
        img = cv2.line(img, 
            tk.tup_float2int(self.top_pt), 
            tk.tup_float2int(self.r_pt), 
            (0,0,0)
        )
        # render the bottom curve of the drop
        cir_center = tuple(np.add((0, self.rad), self.bot_pt))
        lpt_ang = tk.point2rad(cir_center, self.l_pt)
        rpt_ang = tk.point2rad(cir_center, self.r_pt)
        ang_vec = Goutte.gen_largest_arc(lpt_ang, rpt_ang)
        # ang_vec = [lpt_ang, rpt_ang]
        pts = Goutte.rad2point(cir_center, self.rad, ang_vec)
        for pt in pts:
            tk.set_pixel(img, *pt, (0, 0, 0))
        # plot the four main points in red
        if self.debug:  
            tk.render_debug_point(img, self.bot_pt)
            tk.render_debug_point(img, self.top_pt)
            tk.render_debug_point(img, self.l_pt)
            tk.render_debug_point(img, self.r_pt)
            tk.render_debug_point(img, cir_center)

    @staticmethod
    def gen_largest_arc(ang1:float, ang2:float) -> List[float]:
        """generate the largest arc between two angles in radians.
        ang1 is the first angle.
        ang2 is the second angle.
        """
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

    @staticmethod
    def rad2point(
        center: Tuple[float, float], 
        rad: float,
        angles:List[float]
    ) -> List[Tuple[float, float]]:
        """convert a list of angles in radians to a list of points on 
        the ellipse.
        center is the center of the ellipse.
        rad is the radius of the ellipse.
        angles is a list of angles in radians.
        """
        x = []
        y = []
        for angle in angles:
            xi, yi = tk.rad2point(center, rad, angle)
            x.append(xi)
            y.append(yi)
        return list(zip(x, y))

if __name__ == "__main__":
    ang = float(np.deg2rad(45))
    goutte = Goutte(300, ang, (200, 200))
    img = np.ones((400, 400, 3), dtype = "uint8") * 255
    goutte2 = Goutte.scaled_goutte(img, goutte, 0.85)
    goutte.render(img)
    goutte2.render(img)
    num_labels, labels = cv2.connectedComponents(img[:,:,0], connectivity=4)
    plt.imshow(labels, cmap='gray')  # Use 'gray' colormap for grayscale
    plt.colorbar()  # Optional: adds a color scale
    plt.show()
    img = img[::-1,:,:]
    while True:
        cv2.imshow("output", img)
        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            break

    

