from typing import List, Tuple, Optional
from delaunay import Triangulation
import numpy as np



class Refinement:
    def __init__(
        self, 
        verts: List[Tuple[float, float]], 
        segs: List[Tuple[int, int]]
    ):
        self.verts: List[Tuple[float, float]] = verts
        self.segs : List[Tuple[int, int]] = segs
        self.factice_pts = []

    def __call__(self):
        self._add_bounding_square()
        tri = Triangulation(self.verts)
        tri()
        #edges = tri.get_edges()
        # create the combination of pairs of edges
        #edge_comb = itertools.combinations(edges, 2)
        # ? Not sure whether I should also iterate through the segments 
        # ? of the triangution here
        segs_to_check = self.segs
        segs_to_remove = []
        while len(segs_to_check) > 0:
            seg = segs_to_check.pop(0)
            for vert in self.verts:
                if vert != self.verts[seg[0]] and vert != self.verts[seg[1]]:
                    ang = angle(vert, self.verts[seg[0]], self.verts[seg[1]])
                    assert ang >= 0, "handle the case where the angle is negative"
                    if ang < np.pi /2 :
                        # add the center of the segment to the list of vertices
                        new_vert = (
                            (self.verts[seg[0]][0] + self.verts[seg[1]][0]) / 2,
                            (self.verts[seg[0]][1] + self.verts[seg[1]][1]) / 2
                        )
                        self.verts += [new_vert]
                        tri.add_point(new_vert)
                        # add the new segments
                        self.segs.append((seg[0], len(self.verts) - 1))
                        self.segs.append((seg[1], len(self.verts) - 1))
                        # remove the segment from the list of segments
                        segs_to_remove.append(seg)
        pass




    def _add_bounding_square(self):
        # compute the extremes of the points
        min_x = min([p[0] for p in self.verts])
        max_x = max([p[0] for p in self.verts])
        min_y = min([p[1] for p in self.verts])
        max_y = max([p[1] for p in self.verts])
        span = max(max_x - min_x, max_y - min_y)
        # add the bounding square
        mid = ((min_x + max_x) / 2, (min_y + max_y) / 2)
        square_pts = []
        square_pts.append((mid[0] - span, mid[1] - span))
        square_pts.append((mid[0] + span, mid[1] - span))
        square_pts.append((mid[0] + span, mid[1] + span))
        square_pts.append((mid[0] - span, mid[1] + span))
        self.factice_pts += square_pts
        self.verts += square_pts
        # add the segments
        self.segs.append((len(self.verts) - 4, len(self.verts) - 3))
        self.segs.append((len(self.verts) - 3, len(self.verts) - 2))
        self.segs.append((len(self.verts) - 2, len(self.verts) - 1))
        self.segs.append((len(self.verts) - 1, len(self.verts) - 4))
        
def ccw(A,B,C):
    return (C.y-A.y) * (B.x-A.x) > (B.y-A.y) * (C.x-A.x)

# Return true if line segments AB and CD intersect
def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)
        
def angle(sommet:Tuple[float, float], pt1: Tuple[float, float], pt2: Tuple[float, float]):
    """Return the angle between the two vectors (sommet, pt1) and (sommet, pt2)"""
    v1 = (pt1[0]-sommet[0], pt1[1]-sommet[1])
    v2 = (pt2[0]-sommet[0], pt2[1]-sommet[1])
    return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))