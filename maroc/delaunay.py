from typing import List, Tuple
import matplotlib.pyplot as plt
import random
import math

class Triangulation:
    def __init__(self, points: List[Tuple[float]]):
        # check that not two points are the same
        assert len(points) == len(set(points)), "Two points are the same"
        self.points_to_add = points
        self.points = []
        self.factice_points = []
        self.edges: List[Tuple[int]] = [] # Contains the indices of the points
        self.triangles: List[Tuple[int]] = [] # Contains the indices of the edges
        self.tolerance = 1e-6
    
    def _triangle_to_edges(self, triangle: tuple[int]) -> tuple[tuple[int]]:
        """
        Convert a triangle to its edges.
        """
        return ((triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[2], triangle[0]))

    def plot(self):
        """plot the points and the edges of triangles"""
        plt.scatter(*zip(*self.points))
        for tri in self.triangles:
            # plot the three edges of the triangle separately
            for edge in self._triangle_to_edges(tri):
                plt.plot(*zip(*[self.points[edge[0]], self.points[edge[1]]]), color='black')
        plt.show()
    
    def delaunay(self):
        """
        Compute the Delaunay triangulation of the points.
        """
        # create the initial triangulation
        self._make_container_triangle()
        for point in self.points_to_add:
            tri = self.what_triangle(point)
            # add the point to the triangulation
            self.points.append(point)
            point_id = len(self.points) - 1
            edge = self._find_edge_if_one(tri, point)
            if edge == []:
                # add the edges of the triangle to the triangulation
                # init_trln.edges.append((tri[0], point_id))
                # init_trln.edges.append((tri[1], point_id))
                # init_trln.edges.append((tri[2], point_id))
                # add the triangles to the triangulation
                self.triangles.append((tri[0], tri[1], point_id))
                self.triangles.append((tri[1], tri[2], point_id))
                self.triangles.append((tri[2], tri[0], point_id))
                # remove the triangle from the triangulation
                self.triangles.remove(tri)
                # legalize the edges of the triangulation
                self._legalize_edge(point_id, (tri[0], tri[1]))
                self._legalize_edge(point_id, (tri[1], tri[2]))
                self._legalize_edge(point_id, (tri[2], tri[0]))
            else:
                # get the point of the triangle that is not on the edge
                other_point = [p for p in tri if p not in edge][0]
                # add the triangles to the triangulation
                self.triangles.append((edge[0], other_point, point_id))
                self.triangles.append((edge[1], other_point, point_id))
                # remove the triangle from the triangulation
                self.triangles.remove(tri)
                # legalize the edges of the triangulation
                self._legalize_edge(point_id, (edge[0], other_point))
                self._legalize_edge(point_id, (edge[1], other_point))
            #self.plot()
        # remove the triangles that contain a factice point
        to_remove = []
        for tri_id, tri in enumerate(self.triangles):
            for point_id in tri:
                if self.points[point_id] in self.factice_points:
                    print(tri_id)
                    to_remove.append(tri_id)
                    break
        for tri_id in reversed(to_remove):
            self.triangles.pop(tri_id)
        #self.triangles = [tri for tri in self.triangles if not any(p in self.factice_points for p in tri)]
        # remove the factice points from the points
        self.points = [p for p in self.points if p not in self.factice_points]
        # substract all point ids by the number of factice points in the triangles
        self.triangles = [(t[0] - len(self.factice_points), t[1] - len(self.factice_points), t[2] - len(self.factice_points)) for t in self.triangles]

    def _make_container_triangle(self):
        # get the point with the highest y and x coordinates
        # max_point = self._max_point()
        # create two other points such that the triangle formed by the three points
        # contains all the points of the triangulation
        x_min = min(self.points_to_add, key=lambda p: p[0])[0]
        x_max = max(self.points_to_add, key=lambda p: p[0])[0]
        y_min  = min(self.points_to_add, key=lambda p: p[1])[1]
        y_max = max(self.points_to_add, key=lambda p: p[1])[1]
        max_point = (x_max, y_max)
        fict_point_1 = (x_min - 2 * (x_max - x_min), y_max)
        fict_point_2 = (x_max, y_min - 2 * (y_max - y_min))
        #self.points_to_add.remove(max_point)
        # check if max_point is already in the points to add
        if max_point in self.points_to_add:
            self.points_to_add.remove(max_point)
        else:
            self.factice_points.append(max_point)
        self.points.append(max_point)
        self.points.append(fict_point_1)
        self.points.append(fict_point_2)
        self.factice_points.append(fict_point_1)
        self.factice_points.append(fict_point_2)
        self.triangles.append((0, 1, 2))

    def what_triangle(self, point: tuple[float]) -> int:
        """
        Find the triangle that contains the point.
        """
        for tri in self.triangles:
            if self._is_in_triangle(point, tri):
                return tri
        raise ValueError("The point is not in any triangle")
    
    def _is_in_triangle(self, point: tuple[float], triangle: tuple[int]) -> bool:
        """
        Check if the point is in the triangle.
        """
        tri_points = [self.points[i] for i in triangle]
        # compute the area of the triangle
        area = self._area(tri_points)
        # compute the area of the triangles formed by the point and the 
        # vertices of the triangle
        area1 = self._area([point, tri_points[0], tri_points[1]])
        area2 = self._area([point, tri_points[1], tri_points[2]])
        area3 = self._area([point, tri_points[2], tri_points[0]])
        # if the sum of the areas of the triangles formed by the point 
        # and the vertices of the triangle is equal to the area of the 
        # triangle, then the point is in the triangle
        # add a tolerance of order 1e-4
        return abs(area - (area1 + area2 + area3)) < self.tolerance
    
    def _find_edge_if_one(self, triangle: tuple[int], point: tuple[float]) -> tuple[int]:
        """
        Find the edge of the triangle that contains the point.
        """
        for edge in self._triangle_to_edges(triangle):
            if self._is_on_edge(point, edge):
                return edge
        return []


    def _is_on_edge(self, point: tuple[float], edge: tuple[int]) -> bool:
        """
        Check if the point is on the edge.
        """
        edge_points = [self.points[i] for i in edge]
        # compute the area of the triangle formed by the point and the 
        # vertices of the edge
        area = self._area([point, edge_points[0], edge_points[1]])
        # check if the area is equal to 0
        is_on_edge = (area <= self.tolerance)
        return is_on_edge

    def _area(self, points: list[tuple[float]]) -> float:
        """
        Compute the area of the triangle formed by the points.
        """
        return abs(
            (
                points[0][0] * (points[1][1] - points[2][1]) + \
                points[1][0] * (points[2][1] - points[0][1]) + \
                points[2][0] * (points[0][1] - points[1][1])
            ) / 2.0
        )

    def _legalize_edge(self, point_id: int, edge: tuple[int]):
        """
        Legalize the edge of the triangulation.
        """
        # check if the triangle formed by the edge and the point is in the
        # triangulation
        cur_tri = ()
        opp_tri = ()
        for tri in self.triangles:
            if edge[0] in tri and edge[1] in tri and point_id in tri:
                # if the triangle is in the triangulation, then the edge is legal
                cur_tri = tri
            # find the potential other triangle that shares the edge
            if edge[0] in tri and edge[1] in tri and point_id not in tri:
                opp_tri = tri
        assert cur_tri != (), "The triangle formed by the edge and the \
            point is not in the triangulation."
        # there might not be a triangle that contains the edge and not the point
        # In this case, the edge is on the convex hull and the edge is legal
        if opp_tri == ():
            return
        # else, we must check if the edge is legal
        opp_point_id = [p for p in opp_tri if p != edge[0] and p != edge[1]][0]
        if self._is_legal(point_id, edge, opp_point_id):
            return
        else:
            # if the edge is not legal, we must flip it
            self._flip_edge(point_id, edge, opp_point_id)
            # legalize the edges of the new triangles
            self._legalize_edge(point_id, (opp_point_id, edge[0]))
            self._legalize_edge(point_id, (opp_point_id, edge[1]))

    def _flip_edge(self, point_id: int, edge: tuple[int], opp_point_id: int):
        """
        Flip the edge of the triangulation.
        """
        # remove the triangles that contain the edge
        for tri in self.triangles:
            if edge[0] in tri and edge[1] in tri:
                self.triangles.remove(tri)
        # add the new triangles
        self.triangles.append((point_id, edge[0], opp_point_id))
        self.triangles.append((point_id, edge[1], opp_point_id))


    def _is_legal(self, point_id: int, edge: tuple[int], opp_point_id: int) -> bool:
        """
        Check if the edge is legal.
        """
        cur_ang = self._angle(point_id, edge[0], edge[1])
        opp_ang_on_circle = math.pi - cur_ang
        opp_ang = self._angle(opp_point_id, edge[0], edge[1])
        # if the angle formed by the edge and the point is greater than the
        # angle formed by the edge and the other point, then the edge is legal
        return opp_ang_on_circle >= opp_ang

    def _angle(self, sommet: int, point1: int, point2: int) -> float:
        """
        Compute the angle formed by the points.
        """
        x1, y1 = self.points[point1][0] - self.points[sommet][0], \
            self.points[point1][1] - self.points[sommet][1]
        x2, y2 = self.points[point2][0] - self.points[sommet][0], \
            self.points[point2][1] - self.points[sommet][1]
        ang = math.acos(
            (x1 * x2 + y1 * y2) / (
                math.sqrt(x1 ** 2 + y1 ** 2) * \
                math.sqrt(x2 ** 2 + y2 ** 2)
            )
        )
        return ang



if __name__ == "__main__":
    # create a random set of points
    points = [(random.random(), random.random()) for _ in range(10)]
    tri  = Triangulation(points)
    tri.delaunay()
    tri.plot()