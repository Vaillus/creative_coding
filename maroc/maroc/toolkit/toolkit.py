import cv2
import numpy as np
from math import atan2
from typing import Tuple, Union
import matplotlib.colors as mcolors






# === type conversion functions ========================================
def tup_float2int(tup:Tuple[float, ...]) -> Tuple[int, ...]:
    n_tup = tuple([int(x) for x in tup])
    return n_tup



# === rendering functions ==============================================



def set_pixel(
    img: np.ndarray, 
    x: int, 
    y: int, 
    color: Tuple[int]
):
    """Set the color of a pixel in an image, handling the coordinate system difference."""
    if type(x) != float:
        pt = tup_float2int((y, x))
    else:
        pt = (y, x)
    img[pt] = color

def render_debug_point(
    img: np.ndarray, 
    pt:Tuple[int, int], 
    color: Union[Tuple[int, int, int], str] = (0, 0, 255)
):
    pt = tup_float2int(pt)
    if isinstance(color, str):
        color = color_name_to_rgb(color)
    img = cv2.circle(
        img, 
        pt, 
        2, 
        color,
        2
    )

def render_debug_line(
    img: np.ndarray, 
    pt1: Tuple[float, float], 
    pt2: Tuple[float, float], 
    color: Union[Tuple[int, int, int], str] = (0, 0, 255)
):
    """render a line between two points"""
    pt1 = tup_float2int(pt1)
    pt2 = tup_float2int(pt2)
    if isinstance(color, str):
        color = color_name_to_rgb(color)
    img = cv2.line(img, pt1, pt2, color, 2)

def color_name_to_rgb(color_name):
    try:
        rgb = mcolors.to_rgb(color_name)
        # Convert to 0-255 range
        rgb_255 = tuple(int(x * 255) for x in rgb)
        return rgb_255
    except ValueError:
        print(f"Color name '{color_name}' is not recognized.")
        return None



def point2rad(center: Tuple[float, float], pt: Tuple[float, float]) -> float:
    """get a point on the circle and convert it to the radius from the center
    of the ellipse"""
    dx = pt[0] - center[0]
    dy = pt[1] - center[1]
    angle = atan2(dy, dx)
    # Ensure the angle is positive and in the range [0, 2π)
    if angle < 0:
        angle += 2 * np.pi
    return angle

def rad2point(center: Tuple[float, float], rad: float, angle: float) -> Tuple[float, float]:
    x = center[0] + rad * np.cos(angle)
    y = center[1] + rad * np.sin(angle)
    return (x, y)

def interpolate(a,b,frac):
    """Return a value between two values a and b, depending on the 
    fraction given as argument.
    if a = 0, b=1 and frac = 0.1, the result is 0.1.
    if a = 1, b=0 and frac = 0.1, the result is 0.9.

    Args:
        a (float): the first value
        b (float): the second value
        frac (float, optional): the fraction of the difference between 
        a and b.

    Returns:
        float: the middle value
    """
    valmin = min(a,b)
    valmax = max(a,b)
    diff = valmax - valmin
    if a < b:
        return float(valmin) + diff * frac
    else:
        return float(valmax) - diff * frac
    
def interpolate_pts(
        pt1: Tuple[float, ...], 
        pt2: Tuple[float, ...], 
        frac: float
    ) -> Tuple[float, ...]:
    """interpolate 2 points"""
    return tuple([interpolate(pt1[i], pt2[i], frac) for i in range(len(pt1))])
