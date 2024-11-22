from typing import Tuple
import numpy as np

import maroc.toolkit.toolkit as tk

class Petal:
    def __init__(
        self, 
        base:tk.Point3D, 
        tip:tk.Point3D
    ):
        self.base = base
        self.tip = tip
        # Calculate initial r, theta, phi
        self.r, self.theta, self.phi = self._cartesian_to_spherical()
    
    def _cartesian_to_spherical(self) -> Tuple[float, float, float]:
        """Convert current tip position (relative to base) to spherical coordinates."""
        # Get vector from base to tip
        dx = self.tip.x - self.base.x
        dy = self.tip.y - self.base.y
        dz = self.tip.z - self.base.z
        
        # Calculate r (length)
        r = np.sqrt(dx**2 + dy**2 + dz**2)
        
        # Calculate theta (angle in xy plane from x-axis)
        theta = np.arctan2(dy, dx)
        
        # Calculate phi (angle from z-axis)
        phi = np.arccos(dz/r)
        
        return r, theta, phi
    
    def plot(self, ax):
        # plot the base point
        ax.scatter(self.base.x, self.base.y, self.base.z)
        ax.plot([self.base.x, self.tip.x], 
                [self.base.y, self.tip.y], 
                [self.base.z, self.tip.z])
        
    def set_phi(self, new_phi: float):
        """
        Change the angle phi (from z-axis) while keeping r and theta constant.
        new_phi: angle in radians from z-axis
        """
        # Calculate new tip position using spherical coordinates
        dx = self.r * np.sin(new_phi) * np.cos(self.theta)
        dy = self.r * np.sin(new_phi) * np.sin(self.theta)
        dz = self.r * np.cos(new_phi)
        
        # Update tip position
        self.tip.x = self.base.x + dx
        self.tip.y = self.base.y + dy
        self.tip.z = self.base.z + dz
        
        # Update stored phi
        self.phi = new_phi



