from typing import List, Tuple, Optional
from delaunay import Triangulation
import numpy as np
from scipy.spatial import Delaunay



class Refinement:
    def __init__(
        self, 
        verts: List[Tuple[float, float]], 
        segs: List[Tuple[int, int]],
        alpha: float = np.pi / 18
    ):
        self.verts: List[Tuple[float, float]] = verts
        self.segs : List[Tuple[int, int]] = segs
        self.factice_pts = []
        self.alpha = alpha

    def __call__(self):
        self._add_bounding_square()
        # tri = Triangulation(self.verts.copy())
        # tri()
        tri = Delaunay(self.verts.copy(), incremental=True)
        all_clean = False
        while not(all_clean):
            print("start while")
            all_clean = True
            tri = self.refine_outline(tri)
            # iterate on triangles. If the triangle has an angle smaller 
            # than alpha, do the following :
            # - add a point if it does not encroach on any segment
            # - split the segments being encroached it does, but do not 
            #   add the point
            for tr in tri.simplices:
                # check if the minimum angle is below self.alpha
                if tri_min_angle(tr, tri.points) < self.alpha:
                    all_clean = False
                    # create the cirumcenter of the triangle
                    p = tri_circumcenter(tr, tri.points)
                    # check if the point encroaches on the outline. If it does,
                    # split all of the segments that it encroaches on.
                    segs_to_check = self.segs.copy()
                    segs_to_remove = []
                    while len(segs_to_check) > 0:
                        seg = segs_to_check.pop(0)
                        ang = angle(p, self.verts[seg[0]], self.verts[seg[1]])
                        if ang > np.pi / 2:
                            segs_to_check, segs_to_remove, tri = self.split_segment(
                                seg, tri, segs_to_check, segs_to_remove)
                    # if there are no segments to remove, it means that the 
                    # point does not encroach on the outline and we can add it.
                    is_encroached = len(segs_to_remove) > 0
                    for seg in segs_to_remove:
                        self.segs.remove(seg)
                    if not is_encroached:
                        self.verts.append(p)
                        tri.add_points([p])
                    break
        pass

    def _add_bounding_square(self):
        """Compute the extreme points.
        From those points, compute the bounding square.
        Add the bounding square elements to the list of vertices and 
        segments.
        """
        # compute the extremes of the points
        min_x = min([p[0] for p in self.verts])
        max_x = max([p[0] for p in self.verts])
        min_y = min([p[1] for p in self.verts])
        max_y = max([p[1] for p in self.verts])
        span = max(max_x - min_x, max_y - min_y)
        # add the bounding square vertices to the list of vertices
        # and the list of factice vertices
        mid = ((min_x + max_x) / 2, (min_y + max_y) / 2)
        square_pts = []
        square_pts.append((mid[0] - span, mid[1] - span))
        square_pts.append((mid[0] + span, mid[1] - span))
        square_pts.append((mid[0] + span, mid[1] + span))
        square_pts.append((mid[0] - span, mid[1] + span))
        self.factice_pts += square_pts
        self.verts += square_pts
        # add the segments of the bounding square to the list of segments
        self.segs.append((len(self.verts) - 4, len(self.verts) - 3))
        self.segs.append((len(self.verts) - 3, len(self.verts) - 2))
        self.segs.append((len(self.verts) - 2, len(self.verts) - 1))
        self.segs.append((len(self.verts) - 1, len(self.verts) - 4))

    def del_out_points(self, tri: Triangulation):
        # delete the points that are outside the outline
        for pt in tri.points:
            if pt not in self.verts:
                pass
                # tri.del_point(pt)

    def refine_outline(self, tri:Triangulation) -> Triangulation:
        # Create the list of the segments to check for encroachment 
        # (outline only)
        segs_to_check = self.segs.copy()
        segs_to_remove = []
        # iterate over all the segments to check. Some segments may be
        # split, creating new segments to check on the go.
        while len(segs_to_check) > 0:
            seg = segs_to_check.pop(0)
            # check all the vertices. If one of them is inside the 
            # circumcircle, split the segment and go to the next one.
            for vert in self.verts:
                # check that the vertice considered is not one of the two 
                # points of the segment
                if vert != self.verts[seg[0]] and vert != self.verts[seg[1]]:
                    # the vertice is in the circumcircle if the angle 
                    # between it and the two points of the segment is 
                    # below 90 degrees
                    ang = angle(vert, self.verts[seg[0]], self.verts[seg[1]])
                    if ang > np.pi /2 :
                        segs_to_check, segs_to_remove, tri = self.split_segment(
                            seg,
                            tri,
                            segs_to_check,
                            segs_to_remove
                        )
                        break
        # remove the segments that have been split
        for seg in segs_to_remove:
            self.segs.remove(seg)
        return tri

    def split_segment(
        self, 
        seg, 
        tri: Triangulation,
        segs_to_check: List[Tuple[int, int]],
        segs_to_remove: List[Tuple[int, int]]
        ):
        """Split a segment in two. Add the new points to the list of
        vertices. Add the new segments to the list of segments to check.
        Add the new point to the triangulation.
        """
        # generate a point at the center of the segment
        new_vert = (
            (self.verts[seg[0]][0] + self.verts[seg[1]][0]) / 2,
            (self.verts[seg[0]][1] + self.verts[seg[1]][1]) / 2
        )
        # Add the generated point to the list of vertices and to the 
        # triangulation and get its index.
        if new_vert in self.verts:
            raise ValueError("The point is already in the list")
        self.verts += [new_vert]
        tri.add_points([new_vert])
        vert_index = len(self.verts) - 1
        # add the new segments
        self.segs.append((seg[0], vert_index))
        self.segs.append((seg[1], vert_index))
        segs_to_check.append((seg[0], vert_index))
        segs_to_check.append((seg[1], vert_index))
        # remove the segment from the list of segments
        segs_to_remove.append(seg)
        return segs_to_check, segs_to_remove, tri
        
def angle(
    sommet:Tuple[float, float], 
    pt1: Tuple[float, float], 
    pt2: Tuple[float, float]
) -> float:
    """Return the angle between the two vectors (sommet, pt1) and (sommet, pt2)"""
    v1 = (pt1[0]-sommet[0], pt1[1]-sommet[1])
    v2 = (pt2[0]-sommet[0], pt2[1]-sommet[1])
    val = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    # handle the case where the dot product is not in the range [-1, 1]
    if val > 1:
        val = 1
    elif val < -1:
        val = -1
    return np.arccos(val)

def tri_min_angle(tri, pts):
    angles = []
    for i in range(3):
        angles.append(angle(pts[tri[i]], pts[tri[(i+1)%3]], pts[tri[(i+2)%3]]))
    return min(angles)

def tri_circumcenter(tri:Tuple[int, int, int], pts:Tuple[float, float]) -> Tuple[float, float]:
        """
        Get the circumcenter of a triangle. not the baricenter !!! 
        It can be outside the triangle.
        """
        a = pts[tri[0]]
        b = pts[tri[1]]
        c = pts[tri[2]]
        d = 2*(a[0]*(b[1]-c[1]) + b[0]*(c[1]-a[1]) + c[0]*(a[1]-b[1]))
        x = (
            (a[0]**2 + a[1]**2)*(b[1]-c[1]) + 
            (b[0]**2 + b[1]**2)*(c[1]-a[1]) + 
            (c[0]**2 + c[1]**2)*(a[1]-b[1])
        )/d
        y = (
            (a[0]**2 + a[1]**2)*(c[0]-b[0]) + 
            (b[0]**2 + b[1]**2)*(a[0]-c[0]) +
            (c[0]**2 + c[1]**2)*(b[0]-a[0])
        )/d
        return (x, y)