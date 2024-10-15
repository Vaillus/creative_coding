from typing import List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt

from maroc.lampe import losange as lo
from maroc.lampe import ellipse as el
from delaunay_study.delaunay import Triangulation
from delaunay_study.delaunday_refinement import Refinement


class Schema:
    def __init__(self, n_mid_arcs: int):
        self.points : List[float] = []
        self.segments : List[Tuple[int, int]] = []
        self.triangles : List[int, int, int] = []
        self.outline = None
        self.inner_los = None
        self.n_mid_arcs = n_mid_arcs

    def get_list_points(self):
        """ Get the list of points of the schema"""
        # generate the four arcs of the outline
        sw, se, nw, ne = self._init_outline()
        # generate the intermediate arcs and store all the arcs in two lists
        # containing parallel arcs md is for "main droite" and mg for "main gauche"
        # "main droite" would be the arcs that one breaks by doing Brice's trick 
        # with the right hand.
        md, mg = self._create_arcs(sw, se, nw, ne)
        # create the losanges and add the points of each losange to the list
        points = []
        # add the points of the outline
        outline = lo.Losange(sw, se, nw, ne, relative=True, pad=0.05)
        points += outline.get_points(10, outer=True)
        # create the inner losanges and add the points of each losange to the list
        for i in range(len(md)-1):
            for j in range(len(mg)-1):
                losange = lo.Losange(mg[i], md[j], md[j+1], mg[i+1], pad=0.15, has_lines=True)
                points += losange.get_points(10)
        return points
    
    def get_points_edges(self, n_points: int, offset:float) -> Tuple[List[float], List[Tuple[int, int]]]:
        """ Generate points and edges of the schema"""
        assert offset >= 0 and offset <= 1
        sw, se, nw, ne = self._init_outline()
        outline = lo.Losange(nw, ne, sw, se, relative=True, pad=0.07)
        md, mg = self._create_arcs(
            outline.isw, 
            outline.ise, 
            outline.inw, 
            outline.ine, 
            offset
        )
        points, edges = [], []
        out_points, out_edges = outline.get_points_edges(n_points, outer=True)
        points += out_points
        edges += out_edges
        for i in range(len(md)-1):
            for j in range(len(mg)-1):
                losange = lo.Losange(
                    mg[i], 
                    md[j], 
                    md[j+1], 
                    mg[i+1], 
                    relative = True, 
                    pad=0.15, 
                    has_lines=True
                )
                los_points, los_edges = losange.get_points_edges(n_points)
                # add len(points) to each value in los_edges
                los_edges = [tuple([x+len(points) for x in edge]) for edge in los_edges]
                points += los_points
                edges += los_edges
        self.points = points
        self.segments = edges
        #points = self.curve_plane(self.points)
        return points, edges

    def _init_outline(self) -> Tuple[el.Arc, el.Arc, el.Arc, el.Arc]:
        # position variables
        # TODO : put the variables in th class variables
        center = 200.0 # x_position of the center of the schema
        wid = 200.0 # width of the schema
        hei = 300.0 # height of the schema
        t_border = 350.0 # bottom border of the schema
        base = (center, t_border - hei) # base of the schema
        left = (center-int(wid/2), t_border - hei + 100) # left border of the schema
        right = (center+int(wid/2), t_border - hei + 100) # right border of the schema
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

    def _create_arcs(self, sw, se, nw, ne, offset):
        md: List[el.Arc] = [ne]
        mg = [nw]
        # offset = 0.1
        # create the middle arcs
        for i in reversed(range(0, self.n_mid_arcs)):
            if offset == 0 and i == 0:
                continue
            mid = lo.Losange.gen_middle(ne,sw, nw, se, (i+offset)/self.n_mid_arcs)
            md += [mid]
            mid2 = lo.Losange.gen_middle(nw, se, ne,sw, (i+offset)/self.n_mid_arcs)
            mg+= [mid2]
        md += [sw]
        mg += [se]
        return md, mg

    def plot(self):
        assert self.points is not None and self.segments is not None
        plt.plot(
            [p[0] for p in self.points], 
            [p[1] for p in self.points], 
            'o'
        )
        for s in self.segments:
            plt.plot(
                [self.points[s[0]][0], self.points[s[1]][0]], 
                [self.points[s[0]][1], self.points[s[1]][1]]
            )
        plt.show()

    @staticmethod
    def curve_plane(points: List[Tuple[float, float]]):

        points = np.array(points)
        print(points)
        # add a z dimension
        #points = np.c_[points, np.zeros(len(points))]
        center = points.mean(axis=0)
        xmin = points[:,0].min()
        xmax = points[:,0].max()
        b = (xmax - xmin)/2
        a = b/4
        x = points[:,0]
        z = a * np.sqrt(1 - (x - center[0])**2 / b**2)
        # plot z against x
        # set nan values in z to 0
        z[np.isnan(z)] = 0
        # put z in the new column of points
        points[:,2] = z
        # convert points to a list of tuples
        points = [tuple(p) for p in points]
        return points
    
    @staticmethod
    def convert_bmesh_verts_to_points(bmverts):
        points = []
        for v in bmverts:
            points += [(v.co[0], v.co[1], v.co[2])]
        return points

if __name__ == "__main__":
    print("coucou")
    schema = Schema(n_mid_arcs=3)
    vertices, segments = schema.get_points_edges(10, 0.40)
    schema.plot()

    # triangulate the points
    # from scipy.spatial import Delaunay
    #tri = Delaunay(vertices)
    #tri = Triangulation(vertices)

    #tri()
    #tri.plot()
    #curve_plane(vertices)
    
    # plot the points
    # assert len(vertices) == len(set(vertices))
    # points=list(set(points))
    # ===
    # tri = Triangulation(points)
    # tri.delaunay()
    # tri.plot()
    # ===
    # from scipy.spatial import Delaunay
    # tri = Delaunay(vertices)
    # import matplotlib.pyplot as plt
    # vertices = np.array(vertices)
    # vertices = np.array(tri.points)
    # plt.triplot(vertices[:,0], vertices[:,1], tri.simplices)
    # plt.plot(vertices[:,0], vertices[:,1], 'o')
    # plt.show()
    # schema.plot()
    # re = Refinement(vertices, segments)
    # re()

# import matplotlib.pyplot as plt
# vertices = np.array(tri.points)
# vertices = np.array(vertices)
# plt.triplot(vertices[:,0], vertices[:,1], tri.simplices)
# plt.plot(vertices[:,0], vertices[:,1], 'o')
# plt.show()

