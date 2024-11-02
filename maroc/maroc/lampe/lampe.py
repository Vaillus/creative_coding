import cv2
import numpy as np
from typing import Tuple, List
import jax.numpy as jnp
from jax import vmap

from maroc.lampe.arc import Arc
from maroc.lampe.losange import Losange, interpolate
from PIL import Image




def base():
    # Create a list to store the images
    imgs = []
    # Initialize the four external arcs constituting the outline of the
    # figure.
    sw, se, nw, ne = init_borders()
    isw, ise, inw, ine = init_inner_arcs(sw, se, nw, ne, 0.03)
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
        md, mg = init_mid_arcs(isw, ise, inw, ine, offset)
        # max_x, min_x, max_y, min_y = compute_max_arc_args(md + mg)
        losanges = init_losanges(md, mg)
        imgs = render(imgs, sw, se, nw, ne, losanges)
        # imgs = render_parallel(imgs, sw, se, nw, ne, losanges)

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
    sw = Arc.init_with_points(lcenter, top_point=left, bottom_point=base)
    se = Arc.init_with_points(rcenter, top_point=right, bottom_point=base)
    nw = Arc.init_with_points(rcenter, top_point=top, bottom_point=left)
    ne = Arc.init_with_points(lcenter, top_point=top, bottom_point=right)
    return sw, se, nw, ne

def init_inner_arcs(sw, se, nw, ne, offset) -> Tuple[Arc, Arc, Arc, Arc]:
    """initialize the inner arcs."""
    isw = Losange.gen_inner_arc_interpolated(sw, ne, se, nw, offset)
    ise = Losange.gen_inner_arc_interpolated(se, nw, ne, sw, offset)
    inw = Losange.gen_inner_arc_interpolated(nw, se, sw, ne, 1-offset)
    ine = Losange.gen_inner_arc_interpolated(ne, sw, se, nw, 1-offset)
    return isw, ise, inw, ine

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
            losange = Losange(mg[i], md[j], md[j+1], mg[i+1], inner_arc_interpolation=True, pad=0.15, has_lines=True)
            losanges.append(losange)
    return losanges

def render(imgs, sw: Arc, se: Arc, nw: Arc, ne: Arc, losanges: List[Losange]):
    img = np.ones((400, 400, 3), dtype = "uint8") * 255
    sw.render(img, bold=False)
    se.render(img, bold=False)
    nw.render(img, bold=False)
    ne.render(img, bold=False)
    for losange in losanges:
        losange.render(img)
    cv2.imshow("output", img)
    imgs += [Image.fromarray(img)]
    return imgs

def render_parallel(
    imgs, sw: Arc, se: Arc, nw: Arc, ne: Arc, losanges: List[Losange]
):
    """
    I tried to parallelize the rendering of the arcs.
    It is not faster.
    I keep this for posterity.
    """
    img = np.ones((400, 400, 3), dtype = "uint8") * 255
    arcs = [sw, se, nw, ne]
    for losange in losanges:
        arcs += losange.get_arcs()
    params = []
    for arc in arcs:
        params += [arc.get_params()]
    pts = vmap(get_arc_pts)(np.array(params))
    flat_pts = pts.reshape(-1, 2)
    flat_pts = flat_pts[~jnp.isnan(flat_pts).any(axis=-1)]
    # convert to int
    flat_pts = flat_pts.astype(jnp.int32)
    flat_pts = np.unique(flat_pts, axis=0)
    img[flat_pts[:, 0], flat_pts[:, 1]] = (0,0,0)
    cv2.imshow("output", img)
    imgs += [Image.fromarray(img)]
    return imgs

def get_arc_pts(params):
    center = (params[0], params[1])
    a = params[2]
    b = params[3]
    tp = (params[4], params[5])
    bp = (params[6], params[7])
    pts = Arc.get_pixels_vectorized_2(center, a, b, tp, bp)
    return pts
    

if __name__ == "__main__":
    # test()
    base()