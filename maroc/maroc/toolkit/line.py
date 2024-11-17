import numpy as np
from typing import Tuple, Union
import maroc.toolkit.toolkit as tk

class Line:
    def __init__(self, pt1: Tuple[float, float], pt2: Tuple[float, float]):
        self.pt1 = pt1
        self.pt2 = pt2

    def get_length(self):
        return np.sqrt((self.pt2[0] - self.pt1[0])**2 + (self.pt2[1] - self.pt1[1])**2)
    
    def render(
        self,
        img: np.ndarray, 
        color: Union[Tuple[int, int, int], str] = (0, 0, 255),
        width: int = 1
    ) -> np.ndarray:
        if isinstance(color, str):
            color = tk.color_name_to_rgb(color)
        # invert x and y for each point
        pt1 = (self.pt1[1], self.pt1[0])
        pt2 = (self.pt2[1], self.pt2[0])
        pt1 = tk.tup_float2int(pt1)
        pt2 = tk.tup_float2int(pt2)
        # get the points of the line between the two points
        pts = tk.get_pixels_line(pt1, pt2)
        pts = pts[:,::-1]
        if width == 1:
            img[pts[:, 0], pts[:, 1]] = color
        else:
            img = tk.thick_line(img, pts, color, width)
            # bin_buffer = np.zeros_like(img[:,:,0])
            # bin_buffer[pts[:, 0], pts[:, 1]] = 1
            # # Create structuring element (e.g., a disk-shaped kernel)
            # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            # # Perform dilation
            # thick_line_mask = cv2.dilate(bin_buffer.astype(np.uint8), kernel)
            # img[thick_line_mask == 1] = color
            # draw_fractional_thick_line(img, pts, width)
        return img
    
    