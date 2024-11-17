import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
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

    def _init_half_arc(self):
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

    def get_pts_on_goutte(self, n_pts):
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
        line_length = self.goutte.right_line.get_length()
        harc_length = self.half_arc.get_length()
        total_length = line_length + harc_length
        lfrac = line_length / total_length
        afrac = harc_length / total_length
        return lfrac, afrac
    
    def _get_pts_on_line(self, n_pts, lfrac):
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

if __name__ == "__main__":
    fleur = Fleur(100, 0.5, (0.0, 0.0))
    fleur.plot()