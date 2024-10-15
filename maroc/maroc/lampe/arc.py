from __future__ import annotations
from math import atan2
import numpy as np
from typing import Tuple, List, Optional
from shapely.geometry.polygon import LinearRing
import matplotlib.pyplot as plt
import cv2
import jax.numpy as jnp
from jax import jit, vmap

@jit
def _render_jax_core(a, b, center, tp, bp, img, color, x, y):
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
        a: Optional[float]=None, 
        b: Optional[float]=None, 
        top_point: Optional[Tuple[float, float]]=None, 
        bottom_point: Optional[Tuple[float, float]]=None
    ):
        """ An arc is inscribed in an ellipse defined by its center and 
        its two axes a (horizontal) and b (vertical).
        The arc is delimited by the starting point and the ending point.
        
        An arc can be initialized in two ways:
        1. with the center and the two axis values, which are enough to 
           compute the equation of the ellipse.
        2. with the center and two points through which the ellipse 
           passes. The foci are computed from these axis values.
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
        self.a = int(a)
        self.b = int(b)
        self.tp = top_point
        self.bp = bottom_point



    # === init function ========================================



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
    


    # === conversion functions ========================================



    def point2rad(self, pt: Tuple[float]) -> float:
        """ From a point on the ellipse, compute the radius from the center
        of the ellipse."""
        # get the angle of the point from the center of the ellipse
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
        xs, ys = Arc.intersections(rad_points_self, rad_points_other)
        # find the point that is between the two limit points of ell2
        x, y = other._sel_good_point(xs, ys)
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
    def intersections(
        pts_ell1: np.ndarray, 
        pts_ell2: np.ndarray
    ) -> Tuple[List[float], List[float]]:
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
            plt.show()
        return xs, ys

    def _sel_good_point(
        self, 
        xs:List[float], 
        ys:List[float], 
    ) -> Tuple[float]:
        """ Two points are usually found as the intersections of the ellipse in 
        which the arc is inscribed and any other. 
        This function selects the point that is inscribed in the arc.
        """
        # find the min and max x and y of the arc
        xmin = min(self.tp[0], self.bp[0])
        xmax = max(self.tp[0], self.bp[0])
        ymin = min(self.tp[1], self.bp[1])
        ymax = max(self.tp[1], self.bp[1])
        # find the point that is between the two limit points of arc
        sel_x = None
        sel_y = None
        num_points = len(xs)
        for i in range(num_points):
            xi = xs[i]
            yi = ys[i]
            if xi <= xmax and xi >= xmin and yi <= ymax and yi >= ymin:
                sel_x = xi
                sel_y = yi
                break
        # In case no point was found, select the closest one 
        if sel_x is None or sel_y is None:
            # select value closest to x between xmin and xmax
            sel_x = min(xs, key=lambda x:abs(x-xmin))
            # ysel equals the corresponding y value
            sel_y = ys[xs.index(sel_x)]
        assert type(sel_x) is float, "x is not a float"
        return sel_x, sel_y



    # === rendering ====================================================



    def render(
        self, 
        img: jnp.ndarray[int, np.dtype[np.int64]], 
        color: Tuple[int]=(0,0,0), 
        bold: bool=False,
        max_x:int=0,
        min_x:int=0,
        max_y:int=0,
        min_y:int=0
    ) -> None:
        # angles is not good because it might miss some points.
        # The choice will be between vanilla and bresenham.
        # I will try to parallelize one of them later.
        self.render_vectorized(img, color)

    def render_vanilla(
        self, 
        img: np.ndarray[int, np.dtype[np.int64]], 
        color: Tuple[int]=(0,0,0), 
        bold: bool=False
    ) -> None:
        # get the limits of the ellipse
        xmax = max(self.tp[0],self.bp[0])
        xmin = min(self.tp[0],self.bp[0])
        ymax = max(self.tp[1],self.bp[1])
        ymin = min(self.tp[1],self.bp[1])
        # render the ellipse
        for x in range(int(-self.a), int(self.a)): # sweep on the x axis
            # check if the point is within the limits of the arc
            if (x+self.center[0] <= xmax) and (x+self.center[0]>=xmin):
                # compute the y coordinates of the points on the ellipse
                yp = self.b * np.sqrt(1 - (x/self.a)**2)
                ym = - yp
                # check if the points are within the limits of the arc
                if (yp+self.center[1] <= ymax) and (yp+self.center[1]>=ymin):
                    img[int(x+self.center[0]), int(yp+self.center[1])] = color
                    # draw all adjacent pixels
                    self._draw_bold_circle(img, x, yp, color, bold=bold)
                if (ym+self.center[1] <= ymax) and (ym+self.center[1]>=ymin):
                    img[int(x+self.center[0]),int(ym+self.center[1])] = color
                    self._draw_bold_circle(img, x, ym, color, bold=bold)
        # do the same as above but sweep on the y axis
        for y in range(int(-self.b), int(self.b)):
            if (y+self.center[1] <= ymax) and (y+self.center[1]>=ymin):
                xp = self.a * np.sqrt(1 - (y/self.b)**2)
                xm = - xp
                if (xp+self.center[0] <= xmax) and (xp+self.center[0]>=xmin):
                    img[int(xp+self.center[0]), int(y+self.center[1])] = color
                    self._draw_bold_circle(img, xp, y, color, bold=bold)
                if (xm+self.center[0] <= xmax) and (xm+self.center[0]>=xmin):
                    img[int(xm+self.center[0]), int(y+self.center[1])] = color
                    self._draw_bold_circle(img, xm, y, color, bold=bold)

    def render_vectorized(self, img, color=(0,0,0)):
        # Define the range for x and y
        x_range = jnp.arange(-arc.a, arc.a + 1)
        xmax = max(arc.tp[0],arc.bp[0])
        xmin = min(arc.tp[0],arc.bp[0])
        ymax = max(arc.tp[1],arc.bp[1])
        ymin = min(arc.tp[1],arc.bp[1])
        x_in_bounds = (x_range + arc.center[0] <= xmax) & (x_range + arc.center[0] >= xmin)
        x_range = x_range[x_in_bounds]
        y_range = jnp.arange(-arc.b, arc.b + 1)
        y_in_bounds = (y_range + arc.center[1] <= ymax) & (y_range + arc.center[1] >= ymin)
        y_range = y_range[y_in_bounds]

        yp = arc.b * jnp.sqrt(1 - (x_range/arc.a)**2)
        ym = -yp

        xp = arc.a * jnp.sqrt(1 - (y_range/arc.b)**2)
        xm = -xp

        # get the x_range, yp pairs for yp in y_range
        yp_in_bounds = (yp + arc.center[1] <= ymax) & (yp + arc.center[1] >= ymin)
        yp_pairs = jnp.array([x_range, yp]).T[yp_in_bounds]
        # get the x_range, ym pairs for ym in y_range
        ym_in_bounds = (ym + arc.center[1] <= ymax) & (ym + arc.center[1] >= ymin)
        ym_pairs = jnp.array([x_range, ym]).T[ym_in_bounds]

        # get the x_range, yp pairs for yp in y_range
        xp_in_bounds = (xp + arc.center[0] <= xmax) & (xp + arc.center[0] >= xmin)
        xp_pairs = jnp.array([x_range, xp]).T[xp_in_bounds]
        # get the x_range, ym pairs for ym in y_range
        xm_in_bounds = (xm + arc.center[0] <= xmax) & (xm + arc.center[0] >= xmin)
        xm_pairs = jnp.array([x_range, xm]).T[xm_in_bounds]


        # Function to process a single x
        def process_x(x, xmax, xmin, ymax, ymin):
            # Compute y coordinates for this x
            yp = self.b * jnp.sqrt(1 - (x/self.a)**2)
            ym = -yp
            
            # Check if x is within the limits of the arc
            x_in_bounds = (x + self.center[0] <= xmax) & (x + self.center[0] >= xmin)
            
            # Check if y points are within the limits of the arc
            yp_in_bounds = (yp + self.center[1] <= ymax) & (yp + self.center[1] >= ymin)
            ym_in_bounds = (ym + self.center[1] <= ymax) & (ym + self.center[1] >= ymin)
            
            # Combine conditions
            should_color_p = x_in_bounds & yp_in_bounds
            should_color_m = x_in_bounds & ym_in_bounds
            
            return x, yp, ym, should_color_p, should_color_m

        def process_y(y, xmax, xmin, ymax, ymin):
            xp = self.a * jnp.sqrt(1 - (y/self.b)**2)
            xm = -xp

            y_in_bounds = (y + self.center[1] <= ymax) & (y + self.center[1] >= ymin)

            xp_in_bounds = (xp + self.center[0] <= xmax) & (xp + self.center[0] >= xmin)
            xm_in_bounds = (xm + self.center[0] <= xmax) & (xm + self.center[0] >= xmin)

            should_color_p = xp_in_bounds & y_in_bounds
            should_color_m = xm_in_bounds & y_in_bounds

            return y, xp, xm, should_color_p, should_color_m

        x, yp, ym, should_color_p, should_color_m = vmap(process_x)(x_range, xmax, xmin, ymax, ymin)
        # from those variables, create the set of px, py that will be colored



        y, xp, xm, should_color_p2, should_color_m2 = vmap(process_y)(y_range, xmax, xmin, ymax, ymin)

        
        # Update the image
        return img.at[py, px].set(new_colors)

    def _draw_bold_circle(
        self, 
        img:np.ndarray[int, np.dtype[np.int64]], 
        x: int, 
        y: int,
        color: Tuple[int], 
        bold:bool
    ) -> None:
        """ Only way I found to draw big points"""
        if bold:
            cv2.circle(
                img, 
                (int(x+self.center[0]), int(y+self.center[1])), 
                2, 
                color, 
                -1
            )

    def render_angles(
        self, 
        img: np.ndarray[int, np.dtype[np.int64]], 
        color: Tuple[int]=(0,0,0), 
        bold: bool=False
    ) -> None:
        n_points = 500
        angle_tp = self.point2rad(self.tp)
        angle_bp = self.point2rad(self.bp)
        if abs(angle_bp - angle_tp) > np.pi:
            if angle_bp > angle_tp:
                angle_bp -= 2 * np.pi
            else:
                angle_tp -= 2 * np.pi
        angles = np.linspace(angle_tp, angle_bp, n_points)

        # Vectorized point calculation
        x = self.center[0] + self.a * np.cos(angles)
        y = self.center[1] + self.b * np.sin(angles)
         # Round to nearest integer and clip to image boundaries
        x = np.clip(np.round(x).astype(int), 0, img.shape[1] - 1)
        y = np.clip(np.round(y).astype(int), 0, img.shape[0] - 1)
        img[y, x] = color
        # points = [self.rad2point(angle) for angle in angles]
        # for px, py in points:
        #     img[int(py), int(px)] = color
        #     if bold:
        #         cv2.circle(img, (int(px), int(py)), 1, color, -1)

    def render_bresenham(
        self, 
        img: np.ndarray[int, np.dtype[np.int64]], 
        color: Tuple[int]=(0,0,0), 
        bold: bool=False
    ) -> None:
        # Calculate start and end angles
        start_angle = self.point2rad(self.tp)
        end_angle = self.point2rad(self.bp)

        # Ensure start_angle is smaller than end_angle
        if start_angle > end_angle:
            start_angle, end_angle = end_angle, start_angle

        # Bresenham's ellipse algorithm
        a2 = self.a * self.a
        b2 = self.b * self.b
        x = 0
        y = self.b
        px = 0
        py = 2 * a2 * y

        # Plot first set of points
        self._plot_ellipse_points_bresenham(img, x, y, color, bold, start_angle, end_angle)

        # Region 1
        p = b2 - (a2 * self.b) + (0.25 * a2)
        while px < py:
            x += 1
            px += 2 * b2
            if p < 0:
                p += b2 + px
            else:
                y -= 1
                py -= 2 * a2
                p += b2 + px - py
            self._plot_ellipse_points_bresenham(img, x, y, color, bold, start_angle, end_angle)

        # Region 2
        p = (b2 * (x + 0.5) * (x + 0.5)) + (a2 * (y - 1) * (y - 1)) - (a2 * b2)
        while y > 0:
            y -= 1
            py -= 2 * a2
            if p > 0:
                p += a2 - py
            else:
                x += 1
                px += 2 * b2
                p += a2 - py + px
            self._plot_ellipse_points_bresenham(img, x, y, color, bold, start_angle, end_angle)

    def _plot_ellipse_points_bresenham(
        self, img, x, y, color, bold, start_angle, end_angle, max_a, max_b
    ):
        points = [
            (self.center[0] + x, self.center[1] + y),
            (self.center[0] - x, self.center[1] + y),
            (self.center[0] + x, self.center[1] - y),
            (self.center[0] - x, self.center[1] - y)
        ]
        for px, py in points:
            angle = self.point2rad((px, py))
            if start_angle <= angle <= end_angle:
                if 0 <= px < img.shape[1] and 0 <= py < img.shape[0]:
                    img[int(py), int(px)] = color
                    if bold:
                        cv2.circle(img, (int(px), int(py)), 1, color, -1)

    def render_jax(self, img, color=(0,0,0), bold=False, max_x=0, min_x=0, max_y=0, min_y=0):
        # Convert inputs to JAX arrays if they aren't already
        img = jnp.array(img)
        color = jnp.array(color)
        center = jnp.array(self.center)
        tp = jnp.array(self.tp)
        bp = jnp.array(self.bp)
        a = jnp.array(self.a)
        b = jnp.array(self.b)
        x = jnp.arange(min_x, max_x+1) 
        y = jnp.arange(min_y, max_y+1)
        return _render_jax_core(a, b, center, tp, bp, img, color, x, y)

    

    @staticmethod
    @jit
    def _compute_ellipse_points(x, y, a, b, center):
        yp = b * jnp.sqrt(1 - (x/a)**2)
        ym = -yp
        xp = jnp.full_like(y, a) * jnp.sqrt(1 - (y/b)**2)
        xm = -xp
        return jnp.stack([
            jnp.column_stack([
                jnp.full_like(y, x + center[0]), 
                jnp.full_like(y, yp) + center[1]
            ]),
            jnp.column_stack([
                jnp.full_like(y, x + center[0]), 
                jnp.full_like(y, ym) + center[1]
            ]),
            jnp.column_stack([xm + center[0], y + center[1]]),
            jnp.column_stack([xp + center[0], y + center[1]])
        ])



    