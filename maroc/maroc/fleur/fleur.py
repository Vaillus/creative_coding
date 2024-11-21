import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
import math

from maroc.goutte.goutte import Goutte
from maroc.lampe.arc import Arc
import maroc.toolkit.toolkit as tk

class Fleur:
    def __init__(
        self,
        height: float,
        angle: float,
        center: Tuple[float, float]
    ):
        self.goutte = Goutte(height, angle, center)
        self.half_arc = self._init_half_arc()
        self.nervures = []

    def _init_half_arc(self) -> Arc:
        """ Initialize the half-arc from the Goutte. 
        We don't need the other half to make a 3-D model.
        """
        ang_start = self.goutte.arc.ang_start
        ang_end = 3 * np.pi /2.0
        half_arc = Arc(
            center=self.goutte.arc.center,
            a = self.goutte.arc.a,
            b = self.goutte.arc.b,
            ang_start=ang_start,
            ang_end=ang_end
        )
        return half_arc
    


    # === Points on the Goutte =========================================



    def get_pts_on_goutte(self, n_pts:int) -> List[Tuple[float, float]]:
        """Get a set of 2D points coordinates that are equally spaced on 
        the right side of a Goutte.
        """
        up_pts = []
        down_pts = []
        lfrac, afrac = self._get_len_fractions()
        # get line points
        up_pts, reste = self._get_pts_on_line(n_pts, lfrac)
        # get half_arc points
        down_pts = self._get_pts_on_arc(n_pts, afrac, reste)
        # return the concatenation of the two lists of points
        pts = up_pts + down_pts
        return pts
    
    def _get_len_fractions(self):
        """Get the proportion of points that are to be placed on the 
        straight line part of the Goutte and on the arc part of the 
        Goutte."""
        line_length = self.goutte.right_line.get_length()
        harc_length = self.half_arc.get_length()
        total_length = line_length + harc_length
        lfrac = line_length / total_length
        afrac = harc_length / total_length
        return lfrac, afrac
    
    def _get_pts_on_line(
        self, 
        n_pts:int, 
        lfrac:float
    ) -> Tuple[List[Tuple[float, float]], float]:
        """Get the coordinates of points that are on the straight line 
        part of the Goutte. Following lfrac, most of the time, there is 
        a segment fraction remaining on the line after the last point.
        The length of this segment is returned in `reste` such that the 
        first point on the arc can be placed accordingly. 
        """
        n_pts_line_frac = n_pts * lfrac
        n_pts_line = math.floor(n_pts_line_frac)
        reste = n_pts_line_frac - n_pts_line
        pts = []
        for i in range(1, int(n_pts_line_frac)+1):
            frac = i / n_pts_line_frac
            pt = list(tk.interpolate_pts(
                self.goutte.top_pt, 
                self.goutte.r_pt, 
                frac
            ))
            pt[0] = pt[0] - self.goutte.top_pt[0]
            pts.append(pt)
        return pts, reste

    def _get_pts_on_arc(self, n_pts, afrac, reste):
        """"""
        n_pts_harc_frac = n_pts * afrac + reste
        offset = (1.0 - reste)
        n_pts_harc = int(
            np.round(n_pts_harc_frac, 1)
        )
        pts = []
        for i in range(n_pts_harc):
            frac = min((i + offset) / (n_pts_harc_frac), 1.0)
            ang = self.half_arc.interpolate(frac)
            pt = list(self.half_arc.rad2point(ang))
            pt[0] = pt[0] - self.half_arc.center[0]
            pts.append(pt)
        return pts
    


    # === Angles =======================================================



    @staticmethod
    def get_angle(
        h:float, 
        offset=137.5
    ):
        """Get the angle for a given height, following 
        https://algorithmicbotany.org/papers/abop/abop.pdf
        Figure 4.2 p.101"""
        return np.deg2rad(h * offset)
    
    def get_petals_pos(
        self, 
        n_pts:int, 
        offset:float=137.5
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        h = np.arange(n_pts)
        # Calculate points
        angles = Fleur.get_angle(h, offset)
        pts1 = self.get_pts_on_goutte(n_pts)
        pts1 = np.array(pts1)
        # assert pts1.shape == angles.shape
        pts2 = tk.rad2point(
            center=(0.0, 0.0), 
            rad=pts1[:,0], 
            angle=angles
        )
        x, y = pts2
        z = pts1[:,1]
        return x, y, z    
    


    # === Derivatives ==================================================



    def get_derivative(self, x, y, z):
        # TODO : ajouter une vérification sur le fait que le point est 
        # bien sur la goutte.
        x_der = np.cos(self.get_theta(x, y))
        y_der = np.sin(self.get_theta(x,y))
        if z >= self.goutte.arc.center[1]:
            x_der = - x_der
            y_der = - y_der
        z_der = self.get_z_derivative(z)
        z_der = abs(z_der)
        deriv = (
            x_der, 
            y_der,
            z_der
        )
        return deriv
    
    def get_z_derivative(self, z):
        if z >= self.goutte.r_pt[1]:
            z_der = self.get_line_derivative(z)
        else:
            z_der = self.get_harc_derivative(z)
        return z_der

    def get_line_derivative(self, z):
        assert z >= self.goutte.r_pt[1] and z <= self.goutte.top_pt[1], \
            "z must be between the right and the top point of the goutte"
        top_pt = self.goutte.top_pt
        r_pt = self.goutte.r_pt
        return (top_pt[1] - r_pt[1]) / (top_pt[0] - r_pt[0])
    
    def get_harc_derivative(self, z):
        assert z >= self.goutte.bot_pt[1] and z <= self.goutte.r_pt[1], \
            "z must be between the bottom and the right point of the goutte"
        xs = self.half_arc.get_xs_from_y(z)
        # select the one that is right of the center
        x = xs[0] if xs[0] > self.half_arc.center[0] else xs[1]
        ang = self.half_arc.point2rad((x, z))
        deriv = self.half_arc.get_derivative_y(ang)
        return deriv
    
    def get_ang_from_z(self, z):
        assert z >= self.goutte.bot_pt[1] and z <= self.goutte.r_pt[1], \
            "z must be between the bottom and the right point of the goutte"
        


    def get_radius(self, z):
        # depends exclusively on z
        # frac = (z-self.goutte.bot_pt[1]) / self.goutte.hei
        # lfrac, afrac = self._get_len_fractions()
        if z >= self.goutte.r_pt[1]:
            radius = self._get_radius_line(z)
        else:
            radius = self._get_radius_harc(z)
        return radius

    def _get_radius_line(self, z):
        plfrac = (z - self.goutte.r_pt[1]) / self.goutte.right_line.get_length()
        pt = list(tk.interpolate_pts(
            self.goutte.r_pt, 
            self.goutte.top_pt, 
            plfrac
        ))
        pt[0] = pt[0] - self.goutte.top_pt[0]
        radius = pt[0]
        return radius
    
    def _get_radius_harc(self, z):
        z -= self.goutte.arc.center[1]
        radius = self.half_arc.a * np.sqrt(1-(z**2)/self.half_arc.b**2)
        return radius
    
    def get_theta(self, x, y):
        theta = np.arctan2(y, x)
        return theta
        

    # === Plotting ====================================================



    def plot(self, offset=137.5, npts=100, elev=0):
        npts = int(npts)
        # Create new figure each time
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        x, y, z = self.get_petals_pos(npts, offset)
        # Create new scatter plot
        ax.scatter(x, y, z)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('h')
        ax.view_init(elev=elev, azim=45)  # elev is vertical angle, azim is horizontal rotation
        # make x and y axes go in the interval [-1, 1]
        w = self.goutte.hei /2
        ax.set_xlim(-w, w)
        ax.set_ylim(-w, w)
        plt.show()

    def plot_deriv(self, offset=137.5, n_pts=100, line_length=5):
        npts = int(n_pts)
        # Create new figure each time
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        x, y, z = self.get_petals_pos(npts, offset)
        pts = list(zip(x, y, z))
        derivs = [self.get_derivative(*pt) for pt in pts]

        normalized_derivs = []
        for deriv in derivs:
            deriv = tk.normalize_vector(deriv)
            deriv = tuple(deriv * line_length)
            normalized_derivs.append(deriv)

        pts2 = np.add(pts, normalized_derivs)
        # plot the segments
        for i in range(npts):
            ax.plot([pts[i][0], pts2[i][0]], 
                    [pts[i][1], pts2[i][1]], 
                    [pts[i][2], pts2[i][2]])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('h')
        ax.view_init(elev=0, azim=45)  # elev is vertical angle, azim is horizontal rotation
        # make x and y axes go in the interval [-1, 1]
        w = self.goutte.hei /2
        ax.set_xlim(-w, w)
        ax.set_ylim(-w, w)
        plt.show()

def test_deriv_z():
    """ show the derivative of the fleur's profile as a function of z """
    angle = np.deg2rad(60.0)
    fleur = Fleur(height=100, angle=angle, center=(0.0, 0.0))
    zs = np.linspace(fleur.goutte.bot_pt[1], fleur.goutte.top_pt[1], 1000)
    derivs = [fleur.get_z_derivative(z) for z in zs]
    # cap derivs values
    val = 3
    derivs = [min(max(deriv, -val), val) for deriv in derivs]
    plt.title("Derivative of the fleur's profile")
    plt.plot(zs, derivs)
    plt.axvline(x=fleur.goutte.arc.center[1], color='r', linestyle='--')
    plt.axvline(x=fleur.goutte.r_pt[1], color='b', linestyle='--')
    # plt.title("Derivative")
    plt.legend(['derivative', 'center of the arc', 'junction with the line'])
    plt.xlabel("z")
    plt.ylabel("dz")
    plt.show()

def plot_shit():
    angle = np.deg2rad(90.0)
    fleur = Fleur(height=100, angle=angle, center=(0.0, 0.0))
    fleur.plot_deriv(n_pts=246, line_length=7)


if __name__ == "__main__":
    plot_shit()
    # test_deriv_z()