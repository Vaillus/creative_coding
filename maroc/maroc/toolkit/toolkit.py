import cv2
import numpy as np
from math import atan2
from typing import Tuple, Union, List
import matplotlib.colors as mcolors
from PIL import Image

from maroc.toolkit.line import Line






# === type conversion functions ========================================



def tup_float2int(tup:Tuple[float, ...]) -> Tuple[int, ...]:
    n_tup = tuple([int(x) for x in tup])
    return n_tup

def add_points(pt1: Tuple[float, ...], pt2: Tuple[float, ...]) -> Tuple[float, ...]:
    return tuple([pt1[i] + pt2[i] for i in range(len(pt1))])



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

def line(
    img: np.ndarray, 
    pt1: Tuple[float, float], 
    pt2: Tuple[float, float], 
    color: Union[Tuple[int, int, int], str] = (0, 0, 255),
    width: int = 1
) -> np.ndarray:
    return Line(pt1, pt2).render(img, color, width)


def thick_line(
    img: np.ndarray, 
    pts: List[Tuple[float, float]], 
    color: Union[Tuple[int, int, int], str] = (0, 0, 255),
    width: int = 1
):
    bin_buffer = np.zeros_like(img[:,:,0])
    bin_buffer[pts[:, 0], pts[:, 1]] = 1
    # Create structuring element (e.g., a disk-shaped kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (width, width))
    # Perform dilation
    thick_line_mask = cv2.dilate(bin_buffer.astype(np.uint8), kernel)
    img[thick_line_mask == 1] = color
    return img

def triangle(
    img: np.ndarray, 
    pt1: Tuple[float, float], 
    pt2: Tuple[float, float], 
    pt3: Tuple[float, float], 
    color: Union[Tuple[int, int, int], str] = (0, 0, 255),
    width: int = 1
):
    img = Line(pt1, pt2).render(img, color, width)
    img = Line(pt2, pt3).render(img, color, width)
    img = Line(pt3, pt1).render(img, color, width)
    return img

def get_pixels_line(pt1:Tuple[int, int], pt2:Tuple[int, int]):
    points = []
    # compute the equation of the line in the form y = mx + b
    num = pt2[1] - pt1[1]
    den = pt2[0] - pt1[0]
    if den == 0:
        m = 0
    else:
        m = num / den
    b = pt1[1] - m * pt1[0]
    # get points by sweeping the x-axis
    min_x = min(pt1[0], pt2[0])
    max_x = max(pt1[0], pt2[0])
    if min_x != max_x:
        x_range = np.arange(min_x, max_x)        
        ys = m * x_range + b 
        # convert to int
        ys = ys.astype(int)
        points.extend(np.array([x_range, ys]).T)
    # get points by sweeping the y-axis
    min_y = min(pt1[1], pt2[1])
    max_y = max(pt1[1], pt2[1])
    if min_y != max_y:
        y_range = np.arange(min_y, max_y)
        if den == 0:
            # if the line is vertical, all x are the same
            xs = np.full_like(y_range, pt1[0])
        else:
            xs = (y_range - b) / m
            xs = xs.astype(int)
        points.extend(np.array([xs, y_range]).T)
    points = np.unique(points, axis=0)
    return points


def render_debug_point(
    img: np.ndarray, 
    pt:Tuple[int, int], 
    color: Union[Tuple[int, int, int], str] = (0, 0, 255),
    small: bool = False
):
    pt = tup_float2int(pt)
    if isinstance(color, str):
        color = color_name_to_rgb(color)
    pt = (pt[1], pt[0])
    if small:
        set_pixel(img, *pt, color)
    else:
        img = cv2.circle(
            img, 
            2, 
            radius, 
            color,
            2
        )

def draw_bold_circle(
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

def draw_fractional_thick_line_2(
        matrix, 
        line_pixels, 
        thickness
    ):
    """Mais mdr ça dessine un carré autour du point en fait, pas 
    étonnant que ce soit de la merde.
    """
    half_thick = int(thickness // 2)
    for (x, y) in line_pixels:
        # apply the thickness by drawing a small square or circle around each pixel
        for dx in range(-half_thick, half_thick + 1):
            for dy in range(-half_thick, half_thick + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < matrix.shape[0] and 0 <= ny < matrix.shape[1]:
                    matrix[nx, ny] = 1  # set to black or any color you want

def draw_fractional_thick_line(
        matrix, 
        line_pixels, 
        thickness, 
        color=(0, 0, 0)
    ):
    half_thick = thickness / 2.0  # allow fractional thickness
    for (x, y) in line_pixels:
        # expand around each line pixel
        for dx in range(-int(half_thick) - 1, int(half_thick) + 2):
            for dy in range(-int(half_thick) - 1, int(half_thick) + 2):
                nx, ny = x + dx, y + dy
                if 0 <= nx < matrix.shape[0] and 0 <= ny < matrix.shape[1]:
                    # calculate distance from the center of the line pixel
                    distance = np.sqrt(dx**2 + dy**2)
                    # calculate intensity based on distance
                    if distance <= half_thick:
                        intensity = max(0, 1 - (distance / half_thick))
                        # blend each color channel with the specified color and intensity
                        for c in range(3):  # assuming RGB channels
                            matrix[nx, ny, c] = (1 - intensity) * matrix[nx, ny, c] + intensity * color[c]

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

def render(img: np.ndarray):
    # replace x with y and y with x
    img = np.transpose(img, (1, 0, 2))
    img = img[::-1,:,:]
    cv2.imshow("output", img)

def add_frame(imgs, img: np.ndarray):
    img = np.swapaxes(img, 0, 1)[::-1,:,:]
    imgs += [Image.fromarray(img)]
    return imgs

def save_gif(imgs, filename: str):
    imgs[0].save(filename, save_all=True, append_images=imgs[1:], optimize=False, duration=50, loop=0)

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
        val = float(valmin) + diff * frac
    else:
        val = float(valmax) - diff * frac
    assert val <= valmax, "The interpolated value is greater than the maximum value"
    assert val >= valmin, "The interpolated value is lower than the minimum value"
    return val
    
def interpolate_pts(
        pt1: Tuple[float, ...], 
        pt2: Tuple[float, ...], 
        frac: float
    ) -> Tuple[float, ...]:
    """interpolate 2 points"""
    return tuple([interpolate(pt1[i], pt2[i], frac) for i in range(len(pt1))])

def normalize_vector(vec: Tuple[float, ...]) -> Tuple[float, ...]:
    """Regularize a vector to unit length"""
    length = np.sqrt(sum(x**2 for x in vec))
    if length > 0:
        return tuple(x/length for x in vec)
    else:
        return vec

# === region functions =================================================





def flood_fill_mask(
    img: np.ndarray, 
    start_point: Tuple[int, int], 
    connectivity: int = 4
) -> np.ndarray:
    rows, cols = img.shape[:2]
    mask = np.zeros_like(img[:,:,0], dtype=bool)  # Masque de sortie
    stack = [start_point]
    target_value = img[start_point]

    while stack:
        x, y = stack.pop()
        if (mask[x, y] == False) and (img[x, y] == target_value).all():
            mask[x, y] = True  # Marquer le pixel dans le masque
            # Parcourir les voisins en fonction de la connectivité
            neighbors = []
            if connectivity == 4:
                neighbors = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
            elif connectivity == 8:
                neighbors = [(x-1, y-1), (x-1, y), (x-1, y+1), (x, y-1),
                             (x, y+1), (x+1, y-1), (x+1, y), (x+1, y+1)]
            # Ajouter les voisins valides à la pile
            for nx, ny in neighbors:
                if 0 <= nx < rows and 0 <= ny < cols and mask[nx, ny] == False:
                    stack.append((nx, ny))
        
    return mask

