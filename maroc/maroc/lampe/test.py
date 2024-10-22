import cProfile
import numpy as np

from maroc.lampe.arc import Arc
import jax.numpy as jnp

from maroc.lampe.lampe import init_borders, init_mid_arcs, compute_max_arc_args, init_losanges, render


def main():
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
    arc = Arc(lcenter, top_point=left, bottom_point=base)
    img = jnp.ones((400, 400, 3), dtype=np.uint8) * 255
    # img = np.ones((400, 400, 3), dtype=np.uint8) * 255
    arc.render(img)
    # arc.render_vectorizeimgd(img)

def simu_iter():
    imgs = []
    offset = 0.5
    sw, se, nw, ne = init_borders()
    md, mg = init_mid_arcs(sw, se, nw, ne, offset)
    max_x, min_x, max_y, min_y = compute_max_arc_args(md + mg)
    losanges = init_losanges(md, mg)
    imgs = render(imgs, sw, se, nw, ne, losanges, max_x, min_x, max_y, min_y)


if __name__ == "__main__":
    simu_iter()
    # main()
    
    
