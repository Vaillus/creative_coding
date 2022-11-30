import losange as lo
import ellipse as el
from delaunay import Triangulation
from delaunday_refinement import Refinement
from typing import List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt

class Schema:
    def __init__(self):
        self.points : List[float] = []
        self.segments : List[Tuple[int, int]] = []
        self.triangles : List[int, int, int] = []
        self.outline = None
        self.inner_los = None

    def get_list_points(self):
        sw, se, nw, ne = self._init_outline()
        md, mg = self._create_arcs(sw, se, nw, ne)
        
        # create the losanges and add the points of each losange to the list
        points = []
        # add the points of the outline
        outline = lo.Losange(sw, se, nw, ne, relative=True, pad=0.15)
        points += outline.get_points(10)
        # create the inner losanges and add the points of each losange to the list
        for i in range(len(md)-1):
            for j in range(len(mg)-1):
                losange = lo.Losange(mg[i], md[j], md[j+1], mg[i+1], pad=0.15)
                points += losange.get_points(10)
        return points
    
    def get_points_edges(self):
        sw, se, nw, ne = self._init_outline()
        md, mg = self._create_arcs(sw, se, nw, ne)
        points, edges = [], []
        outline = lo.Losange(sw, se, nw, ne, relative=True, pad=0.15)
        out_points, out_edges = outline.get_points_edges(10)
        points += out_points
        edges += out_edges
        for i in range(len(md)-1):
            for j in range(len(mg)-1):
                losange = lo.Losange(mg[i], md[j], md[j+1], mg[i+1], pad=0.15)
                los_points, los_edges = losange.get_points_edges(10)
                # add len(points) to each value in los_edges
                los_edges = [tuple([x+len(points) for x in edge]) for edge in los_edges]
                points += los_points
                edges += los_edges
        return points, edges

    def _create_arcs(self, sw, se, nw, ne):
        n_mid_arcs = 1
        md: List[el.Arc] = [ne]
        mg = [nw]
        offset = 0.1
        # create the middle arcs
        for i in reversed(range(0, n_mid_arcs)):
            if offset == 0 and i == 0:
                continue
            mid = lo.Losange.gen_middle(ne,sw, nw, se, (i+offset)/n_mid_arcs)
            md += [mid]
            mid2 = lo.Losange.gen_middle(nw, se, ne,sw, (i+offset)/n_mid_arcs)
            mg+= [mid2]
        md += [sw]
        mg += [se]
        return md, mg


    def _init_outline(self) -> Tuple[el.Arc, el.Arc, el.Arc, el.Arc]:
        # position variables
        # TODO : put the variables in th class variables
        center = 200.0 # x_position of the center of the schema
        wid = 200.0 # width of the schema
        hei = 300.0 # height of the schema
        t_border = 350.0 # bottom border of the schema
        base = (center, t_border - hei) # base of the schema
        left = (center-int(wid/2), t_border - hei +100) # left border of the schema
        right = (center+int(wid/2), t_border - hei +100) # right border of the schema
        top = (center, t_border) # top border of the schema
        lcenter = (100.0, 10.0)
        rcenter = (300.0, 10.0)
        # # plot the above variables
        # y and x lim to 0
        # plt.xlim(0, 400)
        # plt.ylim(0, 400)
        # plt.show()
        # initialize borders
        sw = el.Arc(lcenter, top_point=left, bottom_point=base)
        se = el.Arc(rcenter, top_point=right, bottom_point=base)
        nw = el.Arc(rcenter, top_point=top, bottom_point=left)
        ne = el.Arc(lcenter, top_point=top, bottom_point=right)
        return sw, se, nw, ne




if __name__ == "__main__":
    print("coucou")
    schema = Schema()
    vertices, segments = schema.get_points_edges()
    # plot the points
    assert len(vertices) == len(set(vertices))
    #points=list(set(points))
    # ===
    # tri = Triangulation(points)
    # tri.delaunay()
    # tri.plot()
    # ===
    re = Refinement(vertices, segments)
    re()

