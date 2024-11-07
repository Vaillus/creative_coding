from __future__ import annotations
from math import atan2
import numpy as np
from typing import Tuple, List, Optional, Union
from shapely.geometry.polygon import LinearRing
import matplotlib.pyplot as plt
import cv2
import jax.numpy as jnp
from jax import jit, vmap
import math
import maroc.toolkit.toolkit as tk

@jit
def _render_jax_core(a, b, center, tp, bp, img, color, x, y):
    """Un reliquat de quand j'essayais jax pour rendre un arc.
    """
    # Step 1: Determine the bounding box of the arc
    xmax, xmin = jnp.maximum(tp[0], bp[0]), jnp.minimum(tp[0], bp[0])
    ymax, ymin = jnp.maximum(tp[1], bp[1]), jnp.minimum(tp[1], bp[1])

    # Step 2: Create fixed-size coordinate arrays for the ellipse
    # x = x - a + center[0]
    # y = y - b + center[1]

    # Step 3: Create a 2D grid of coordinates
    X, Y = jnp.meshgrid(x, y)

    # Step 4: Create masks for different conditions
    x_in_bounds = (X >= xmin) & (X <= xmax)
    y_in_bounds = (Y >= ymin) & (Y <= ymax)
    in_ellipse = (((X - center[0]) / a)**2 + ((Y - center[1]) / b)**2 == 1)

    # Step 5: Combine masks
    mask = x_in_bounds & y_in_bounds & in_ellipse

    # Step 6: Use the mask to update the image
    img.at[Y, X].set(jnp.where(mask, color, img[Y, X]))

class Arc():
    def __init__(
        self, 
        center: Tuple[float, float], 
        a: float, 
        b: float, 
        # top_point: Optional[Tuple[float, float]]=None, 
        # bottom_point: Optional[Tuple[float, float]]=None,
        ang_start:Optional[float]=None,
        ang_end:Optional[float]=None,
        
    ):
        """ An arc is inscribed in an ellipse defined by its center and 
        its two axes a (horizontal) and b (vertical).
        The arc is delimited by the starting and ending angles.
        """
        self.center = center
        self.a = a
        self.b = b
        self.ang_start = ang_start
        self.ang_end = ang_end



    # === init function ================================================



    @staticmethod
    def compute_ellipse_axis(
        center:Tuple[float], 
        p1:Tuple[float], 
        p2:Tuple[float]
    ) -> Tuple[float, float]:
        """ Compute the axis of the ellipse whose center is given 
        and two points through which the ellipse passes.
        """
        Ax = (p1[0]- center[0]) ** 2
        Ay = (p1[1]- center[1]) ** 2
        Bx = (p2[0]- center[0]) ** 2
        By = (p2[1]- center[1]) ** 2
        num = Ay - By
        den = Bx - Ax
        # handle exceptions
        o = float(num)/float(den)
        a = np.sqrt(float(Ax) + float(Ay)/o)
        b = np.sqrt(float(Ax)* o + float(Ay))
        return a, b
    
    @staticmethod
    def init_with_points(
        center, top_point, bottom_point, a=None, b=None
    ):
        """ Fonction dont le but est de rendre la classe compatible avec 
        ce que j'avais fait avec la lampe. Ici seulement deux points de 
        l'ellipse sont nécessaires pour définir l'arc.
        """
        if a is None or b is None:
            a, b = Arc.compute_ellipse_axis(center, top_point, bottom_point)
        arc = Arc(center, a, b)
        ang1 = arc.point2rad(top_point)
        ang2 = arc.point2rad(bottom_point)
        arc.ang_start, arc.ang_end = Arc.shortest_clockwise_arc(ang1, ang2)
        return arc

    @staticmethod
    def shortest_clockwise_arc(a1, a2):
        """ Given two angles, order them such that the shortest clockwise 
        arc is chosen.
        """
        # Normalize angles to be within 0 and 2*pi
        a1 = a1 % (2 * math.pi)
        a2 = a2 % (2 * math.pi)

        # calculate the clockwise difference
        diff = (a2 - a1) % (2 * math.pi)
        
        # if diff > pi, it means the shortest arc is actually counterclockwise
        if diff > math.pi:
            ang_start, ang_end = a2, a1
        else:
            ang_start, ang_end = a1, a2
        return ang_start, ang_end
    
    def point_in_arc(self, pt:Tuple[float]) -> bool:
        """ Check if a point is inside the arc. """
        ang = self.point2rad(pt)
        return self.ang_in_arc(ang)
        

    def ang_in_arc(
            self, 
            ang:Union[float, np.ndarray[int, np.dtype[np.float64]]]
        ) -> Union[bool, np.ndarray[int, np.dtype[np.bool_]]]:
        """ Check if an angle is inside the arc. """
        self.ang_start = self.ang_start % (2 * math.pi)
        self.ang_end = self.ang_end % (2 * math.pi)
        ang = ang % (2 * math.pi)

        if self.ang_start <= self.ang_end:
            # Simple case: ang_start is less than ang_end
            if isinstance(ang, np.ndarray):
                return (self.ang_start <= ang) & (ang <= self.ang_end)
            else:
                return self.ang_start <= ang <= self.ang_end
        else:
            # The arc wraps around 2*pi
            if isinstance(ang, np.ndarray):
                return (self.ang_start <= ang) | (ang <= self.ang_end)
            else:
                return self.ang_start <= ang or ang <= self.ang_end


    # === conversion functions =========================================



    def point2rad(self, pt: Union[Tuple[float], np.ndarray[int, np.dtype[np.float64]]]) -> float:
        """ From a point on the ellipse, compute the radius from the center
        of the ellipse."""
        # get the angle of the point from the center of the ellipse
        if isinstance(pt, np.ndarray):
            if pt.ndim == 1:
                angle = np.atan2((pt[1]-self.center[1])/self.b, (pt[0]-self.center[0])/self.a)
            else:
                angle = np.atan2((pt[:,1]-self.center[1])/self.b, (pt[:,0]-self.center[0])/self.a)
        else:
            angle = atan2((pt[1]-self.center[1])/self.b, (pt[0]-self.center[0])/self.a)
        return angle

    def rad2point(self, angle:float) -> Tuple[float, float]:
        """convert a list of angles in radians to a list of points on the ellipse
        """
        x = self.center[0] + self.a * np.cos(angle)
        y = self.center[1] + self.b * np.sin(angle)
        return x, y
    


    # === intersection functions ========================================



    def intersect(self, other:Arc) -> Tuple[float, float]:
        """compute intersection between the arc and the one
        provided as argument.
        """
        # convert the ellipses into lists of points
        rad_points_self = self.generate_ellipse_points()
        rad_points_other = other.generate_ellipse_points()
        # find the intersection points between the two ellipses
        xs, ys = Arc._intersections(rad_points_self, rad_points_other)
        # plot the points
        # plt.plot(xs, ys, 'ro')
        # plt.show()
        angs = [other.point2rad((x, y)) for x, y in zip(xs, ys)]
        # find the point that is between the two limit points of ell2
        ang = other._sel_good_ang(angs)
        # in case no intersection was found, plot the necessary elements
        # to understand what's going on.
        if ang is None:
            plt.plot(xs, ys, 'ro')
            points = [self.rad2point(a) for a in rad_points_self]
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            plt.plot(rad_points_self[:,0], rad_points_self[:,1])  
            plt.plot(rad_points_other[:,0], rad_points_other[:,1])     
            plt.plot(self.rad2point(self.ang_start)[0], self.rad2point(self.ang_start)[1], 'go')
            plt.plot(self.rad2point(self.ang_end)[0], self.rad2point(self.ang_end)[1], 'go')     
            plt.plot(other.rad2point(other.ang_start)[0], other.rad2point(other.ang_start)[1], 'go')
            plt.plot(other.rad2point(other.ang_end)[0], other.rad2point(other.ang_end)[1], 'go')     
            plt.show()
        x, y = other.rad2point(ang)
        return x, y

    def generate_ellipse_points(
        self,
        inclination:float=0.0,
        n:int=100
    ) -> np.ndarray[int, np.dtype[np.float64]]:
        """generate a list of points from an ellipse in which the arc 
        is inscribed.
        The points are evenly spaced around the ellipse.
        """
        t = np.linspace(0, 2*np.pi, n, endpoint=False)
        st = np.sin(t)
        ct = np.cos(t)
        inclination = np.deg2rad(inclination)
        si = np.sin(inclination)
        ci = np.cos(inclination)
        result = np.empty((n, 2))
        result[:, 0] = self.center[0] + self.a * ci * ct - self.b * si * st
        result[:, 1] = self.center[1] + self.a * si * ct + self.b * ci * st
        return result

    @staticmethod
    def _intersections(
        pts_ell1: np.ndarray, 
        pts_ell2: np.ndarray
    ) -> Tuple[List[float], List[float]]:
        """ From the two lists of points obtained from two Arcs, creates 
        two "linar rings" which are used to compute the intersections 
        between the ellipses in which the arcs are inscribed.
        """
        # convert the points to a LinearRing
        lr1 = LinearRing(pts_ell1)
        lr2 = LinearRing(pts_ell2)
        # find the intersection points
        intersections = lr1.intersection(lr2)
        # extract the x and y coordinates from the intersection points
        try:
            xs = [float(p.x) for p in intersections.geoms]
            ys = [float(p.y) for p in intersections.geoms]
        except:
            print("no intersection")
            # in case of no intersection, plot the ellipses to see what's going on
            plt.plot(pts_ell1[:,0], pts_ell1[:,1])
            plt.plot(pts_ell2[:,0], pts_ell2[:,1])
            plt.plot(xs, ys, 'ro')
            plt.show()
        return xs, ys

    def _sel_good_ang(
        self, 
        angs:List[float]
    ) -> Tuple[float]:
        """ Two points are usually found as the intersections of the ellipse in 
        which the arc is inscribed and any other. 
        This function selects the point that is inscribed in the arc.
        """
        # find the min and max x and y of the arc
        # xmin = min(self.tp[0], self.bp[0])
        # xmax = max(self.tp[0], self.bp[0])
        # ymin = min(self.tp[1], self.bp[1])
        # ymax = max(self.tp[1], self.bp[1])
        # find the point that is between the two limit points of arc
        # sel_x = None
        # sel_y = None
        sel_ang = None
        for ang in angs:
            # self.point_in_arc(pt)
            if self.ang_in_arc(ang):
                sel_ang = ang
        if sel_ang is None:
            # get the closest angle to the start or end of the arc
            min_diff_start = min([abs(ang - self.ang_start) for ang in angs])
            min_diff_end = min([abs(ang - self.ang_end) for ang in angs])
            if min_diff_start < min_diff_end:
                sel_ang = self.ang_start
            else:
                sel_ang = self.ang_end
        return sel_ang



    # === rendering ====================================================



    def render(
        self, 
        img: jnp.ndarray[int, np.dtype[np.int64]], 
        color: Tuple[int]=(0,0,0), 
        width: int=1
    ) -> None:
        # angles is not good because it might miss some points.
        # The choice will be between vanilla and bresenham.
        # I will try to parallelize one of them later.
        # self.render_vanilla(img, color)
        self.render_vectorized(img, color, width)

    def render_vectorized(self, img, color=(0,0,0), width=1):
        """Affiche les points de l'arc à partir des coordonnées 
        calculées de manière vectorisée.
        """
        pts = self.get_pixels_vectorized_ang()
        # remove duplicates rows
        pts = pts.astype(np.int32)
        pts = np.unique(pts, axis=0)
        # filter out points that are outside the image
        pts = pts[(pts[:, 0] < img.shape[0]) & (pts[:, 1] < img.shape[1])]
        pts = pts[(pts[:, 0] >= 0) & (pts[:, 1] >= 0)]
        if width == 1:
            img[pts[:, 0], pts[:, 1]] = color
        else:
            tk.draw_fractional_thick_line(img, pts, width)
            # for x, y in pts:
            #     self._draw_bold_circle(img, y, x, color, width)
    
    def get_pixels_vectorized_ang(self):
        x_range = np.arange(-int(self.a), int(self.a) + 1)
        yp = self.b * np.sqrt(1 - (x_range/self.a)**2)
        yp_pts = np.array([x_range, yp]).T + self.center
        yp_in_bounds = self.point_in_arc(yp_pts)
        # yp_in_bounds = [self.point_in_arc(pt) for pt in yp_pts]
        yp_pts = yp_pts[yp_in_bounds]
        ym = -yp
        ym_pts = np.array([x_range, ym]).T + self.center
        ym_in_bounds = self.point_in_arc(ym_pts)
        # ym_in_bounds = [self.point_in_arc(pt) for pt in ym_pts]
        ym_pts = ym_pts[ym_in_bounds]

        y_range = np.arange(-int(self.b), int(self.b) + 1)
        xp = self.a * np.sqrt(1 - (y_range/self.b)**2)
        xp_pts = np.array([xp, y_range]).T + self.center
        xp_in_bounds = self.point_in_arc(xp_pts)
        # xp_in_bounds = [self.point_in_arc(pt) for pt in xp_pts]
        xp_pts = xp_pts[xp_in_bounds]
        xm = -xp
        xm_pts = np.array([xm, y_range]).T + self.center
        xm_in_bounds = self.point_in_arc(xm_pts)
        # xm_in_bounds = [self.point_in_arc(pt) for pt in xm_pts]
        xm_pts = xm_pts[xm_in_bounds]

        pts = np.concatenate([yp_pts, ym_pts, xp_pts, xm_pts])
        return pts



    def _draw_bold_circle(
        self, 
        img:np.ndarray[int, np.dtype[np.int64]], 
        x: int, 
        y: int,
        color: Tuple[int], 
        width: int
    ) -> None:
        """ Only way I found to draw big points"""
        if width > 1:
            cv2.circle(
                img, 
                (x, y), 
                width, 
                color, 
                0
            )

    def get_params(self):
        """ Récupération des paramètres de l'arc. A été prévue pour afficher 
        tous les arcs de la figure Lampe en même temps mais finalement c'est 
        plus lent que de les afficher séquentiellement.
        """
        return [
            self.center[0], self.center[1], 
            self.a, 
            self.b, 
            # self.tp[0], self.tp[1], 
            # self.bp[0], self.bp[1]
            self.ang_start, self.ang_end
        ]

    @staticmethod
    def get_pixels_vectorized_2(center, a, b, tp, bp):
        """ Méthode pour récupérer les points à afficher.
        Utilise la vectorisation avec Jax. A été prévue pour afficher 
        tous les arcs de la figure Lampe en même temps mais finalement c'est 
        plus lent que de les afficher séquentiellement.
        Par conséquent, cette méthode n'est plus utilisée mais je la 
        laisse pour le cas où j'aurais besoin de la réutiliser.
        """
        # Define the bounds for x and y
        xmax = jnp.max(jnp.array([tp[0],bp[0]]))
        xmin = jnp.min(jnp.array([tp[0],bp[0]]))
        ymax = jnp.max(jnp.array([tp[1],bp[1]])) 
        ymin = jnp.min(jnp.array([tp[1],bp[1]]))

        x_range = jnp.arange(-400, 400)
        x_in_bounds = (x_range + center[0] <= xmax) & (x_range + center[0] >= xmin)
        # x_range = x_range[x_in_bounds]
        x_range = jnp.where(x_in_bounds, x_range, jnp.nan)
        # x_range = x_range[jnp.isfinite(x_range)]
        yp = b * jnp.sqrt(1 - (x_range/a)**2)
        # get the x_range, yp pairs for yp in y_range
        yp_in_bounds = (yp + center[1] <= ymax) & (yp + center[1] >= ymin)
        yp_pts = jnp.array([x_range, yp])
        yp_pts = jnp.where(yp_in_bounds, yp_pts, jnp.nan).T
        ym = -yp
        # get the x_range, ym pairs for ym in y_range
        ym_in_bounds = (ym + center[1] <= ymax) & (ym + center[1] >= ymin)
        ym_pts = jnp.array([x_range, ym])
        ym_pts = jnp.where(ym_in_bounds, ym_pts, jnp.nan).T
        
        y_range = jnp.arange(-400, 400)
        y_in_bounds = (y_range + center[1] <= ymax) & (y_range + center[1] >= ymin)
        y_range = jnp.where(y_in_bounds, y_range, jnp.nan)

        xp = a * jnp.sqrt(1 - (y_range/b)**2)
        # get the x_range, yp pairs for yp in y_range
        xp_in_bounds = (xp + center[0] <= xmax) & (xp + center[0] >= xmin)
        xp_pts = jnp.array([xp, y_range])
        xp_pts = jnp.where(xp_in_bounds, xp_pts, jnp.nan).T

        xm = -xp
        # get the x_range, ym pairs for ym in y_range
        xm_in_bounds = (xm + center[0] <= xmax) & (xm + center[0] >= xmin)
        xm_pts = jnp.array([xm, y_range])
        xm_pts = jnp.where(xm_in_bounds, xm_pts, jnp.nan).T

        # merge the content of yp_pairs, ym_pairs, xp_pairs, xm_pairs such that there are no duplicate points
        pts = jnp.concatenate([yp_pts, ym_pts, xp_pts, xm_pts])
        # add the center point to the merged_pairs
        pts += jnp.array(center)
        # pts = pts.astype(jnp.int32)

        return pts

    def plot(self):
        pts = self.generate_ellipse_points()
        plt.plot(pts[:,0], pts[:,1])
        # plt.show()
    

if __name__ == "__main__":
    arc = Arc((0,0), 1, 1)
    pass

