import losange as lo
import ellipse as el
from delaunay import Triangulation
from typing import List, Tuple, Optional
import numpy as np

class Schema:
    def __init__(self):
        self.points : List[float] = []
        self.triangles : List[int]
        self.outline = None
        self.inner_los = None

    def get_list_points(self):
        sw, se, nw, ne = self._init_outline()
        outline = lo.Losange(sw, se, nw, ne, relative=True, pad=0.1)
        n_mid_arcs = 4
        md: List[el.Arc] = [ne]
        mg = [nw]
        offset = 0
        for i in reversed(range(0, n_mid_arcs)):
            if offset == 0 and i == 0:
                continue
            mid = lo.Losange.gen_middle(ne,sw, nw, se, (i+offset)/n_mid_arcs)
            md += [mid]
            mid2 = lo.Losange.gen_middle(nw, se, ne,sw, (i+offset)/n_mid_arcs)
            mg+= [mid2]
        md += [sw]
        mg += [se]
        points = []
        for i in range(len(md)-1):
            for j in range(len(md)-1):
                losange = lo.Losange(mg[i], md[j], md[j+1], mg[i+1], pad=0.15)
                points += losange.get_points()
        big_losange = lo.Losange(nw, ne, se, sw, pad=0.15)
        points += big_losange.get_points()
        return points


    def _init_outline(self) -> Tuple[el.Arc]:
        # position variables
        # TODO : put the variables in th class variables
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
        sw = el.Arc(lcenter, top_point=left, bottom_point=base)
        se = el.Arc(rcenter, top_point=right, bottom_point=base)
        nw = el.Arc(rcenter, top_point=top, bottom_point=left)
        ne = el.Arc(lcenter, top_point=top, bottom_point=right)
        return sw, se, nw, ne

    