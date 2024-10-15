import cv2
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry.polygon import LinearRing
from typing import Tuple, List

from maroc.lampe.arc import Arc
from maroc.lampe.losange import Losange, interpolate
from PIL import Image


def base():
    # Create a list to store the images
    imgs = []
    # Initialize the four external arcs constituting the outline of the
    # figure.
    sw, se, nw, ne = init_borders()
    # Set the total number of frames
    T = 50
    # Initialize the current frame counter
    t = 0
    while t < T:
        # increment the counter cyclically and set the offset
        if t == T:
            t = 0
        offset = t/T
        t += 1
        md, mg = init_mid_arcs(sw, se, nw, ne, offset)
        max_x, min_x, max_y, min_y = compute_max_arc_args(md + mg)
        losanges = init_losanges(md, mg)
        imgs = render(imgs, sw, se, nw, ne, losanges, max_x, min_x, max_y, min_y)

        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            break
    imgs[0].save('test.gif', save_all=True, append_images=imgs[1:], optimize=False, duration=50, loop=0)

def init_borders() -> Tuple[Arc, Arc, Arc, Arc]:
    """Initialize the four external arcs constituting the outline of the 
    figure.

    Returns:
        Tuple[Arc, Arc, Arc, Arc]: the four external arcs
    """
    # position variables
    center = 200
    wid = 200
    hei = 300
    b_border = 350
    base = (b_border, center)
    left = (b_border -100, center-int(wid/2))
    right = (b_border -100, center+int(wid/2))
    top = (b_border - hei, center)
    lcenter = (390, 100)
    rcenter = (390, 300)
    # initialize borders
    sw = Arc(lcenter, top_point=left, bottom_point=base)
    se = Arc(rcenter, top_point=right, bottom_point=base)
    nw = Arc(rcenter, top_point=top, bottom_point=left)
    ne = Arc(lcenter, top_point=top, bottom_point=right)
    return sw, se, nw, ne

def init_mid_arcs(sw, se, nw, ne, offset):
    """initialize the middle arcs. 
    We stack the arcs parallel to each other in the same list. 
    md is the list containing those facing north-east, and 
    mg is the list containing those facing north-west."""
    n_mid_arcs = 4
    md = [ne]
    mg = [nw]
    # generate the middle arcs
    for i in reversed(range(0, n_mid_arcs)):
        # skip the first arc if offset is 0
        if offset == 0 and i == 0:
            continue
        mid = Losange.gen_inner_arc_interpolated(
            ne, sw, nw, se, (i+offset)/n_mid_arcs
        )
        md += [mid]
        mid2 = Losange.gen_inner_arc_interpolated(
            nw, se, ne, sw, (i+offset)/n_mid_arcs
        )
        mg+= [mid2]
    md += [sw]
    mg += [se]
    return md, mg

def compute_max_arc_args(arcs: List[Arc]) -> Tuple[float, float, float, float]:
    """
    Compute the maximum and minimum values of 'x' and 'y' arguments separately for 
    all arcs in the list. This is done to later parallelize the rendering
    of the arcs with jax.
    Args:
        arcs (List[Arc]): List of Arc objects
    
    Returns:
        Tuple[float, float, float, float]: Maximum and minimum values of 'x' and 'y' arguments
    """
    max_x = float('-inf')
    min_x = float('inf')
    max_y = float('-inf')
    min_y = float('inf')
    for arc in arcs:
        max_x = max(max_x, arc.tp[0], arc.bp[0])
        min_x = min(min_x, arc.tp[0], arc.bp[0])
        max_y = max(max_y, arc.tp[1], arc.bp[1])
        min_y = min(min_y, arc.tp[1], arc.bp[1])
    return max_x, min_x, max_y, min_y

def init_losanges(md, mg):
    n_rows = len(md)
    losanges = []
    for i in range(n_rows-1):
        for j in range(n_rows-1):
            losange = Losange(mg[i], md[j], md[j+1], mg[i+1], 0.15, has_lines=True)
            losanges.append(losange)
    return losanges

def render(imgs, sw: Arc, se: Arc, nw: Arc, ne: Arc, losanges: List[Losange], max_x:int, min_x:int, max_y:int, min_y:int):
        img = np.ones((400, 400, 3), dtype = "uint8") * 255
        sw.render(img, max_x=max_x, min_x=min_x, max_y=max_y, min_y=min_y)
        se.render(img, max_x=max_x, min_x=min_x, max_y=max_y, min_y=min_y)
        nw.render(img, max_x=max_x, min_x=min_x, max_y=max_y, min_y=min_y)
        ne.render(img, max_x=max_x, min_x=min_x, max_y=max_y, min_y=min_y)
        for losange in losanges:
            losange.render(img, max_x=max_x, min_x=min_x, max_y=max_y, min_y=min_y)
        cv2.imshow("output", img)
        imgs += [Image.fromarray(img)]
        return imgs

if __name__ == "__main__":
    # test()
    base()