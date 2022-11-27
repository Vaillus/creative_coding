import losange as lo
import ellipse as el
from delaunay import Triangulation
from typing import List, Tuple, Optional
from shapely.geometry.polygon import LinearRing
import numpy as np

class Schema:
    def __init__(self):
        self.points : List[float] = []
        self.triangles : List[int]

    def get_list_points(self):
        sw, se, nw, ne = self._init_outline()
        n_mid_arcs = 4
        md: List[el.Arc] = [ne]
        mg = [nw]
        offset = 0
        for i in reversed(range(0, n_mid_arcs)):
            if offset == 0 and i == 0:
                continue
            mid = self._gen_middle(ne,sw, nw, se, (i+offset)/n_mid_arcs)
            #mid.render(img)
            md += [mid]
            mid2 = self._gen_middle(nw, se, ne,sw, (i+offset)/n_mid_arcs)
            #mid2.render(img)
            mg+= [mid2]
        md += [sw]
        mg += [se]





    # === main parts creation  =========================================





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

    def _gen_middle(
        self, 
        top_para:el.Arc, 
        bot_para:el.Arc, 
        top_ort:el.Arc, 
        bot_ort:el.Arc, 
        multi:float=0.5
    ) -> el.Arc:
        x = self._mid_val(top_para.center[0], bot_para.center[0], multi)
        y = self._mid_val(top_para.center[1], bot_para.center[1], multi)
        center = (x,y)
        a = self._mid_val(top_para.a, bot_para.a, multi)
        b = self._mid_val(top_para.b, bot_para.b, multi)
        new_ellipse = el.Arc(center, a=a, b=b)
        x, y = self._find_ellipses_intersection(new_ellipse, top_ort)
        new_ellipse.tp = (x,y)
        x, y = self._find_ellipses_intersection(new_ellipse, bot_ort)
        new_ellipse.bp = (x,y)
        return new_ellipse
    
    def _mid_val(self, a, b, multi=0.5) -> float:
        valmin = min(a,b)
        valmax = max(a,b)
        return float(valmin) + (valmax - valmin) * multi

    