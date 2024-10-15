from typing import List, Tuple
import matplotlib.pyplot as plt
import random
import math

class Triangulation:
    def __init__(self, points: List[Tuple[float, float]]):
        # check that not two points are the same
        assert len(points) == len(set(points)), "Two points are the same"
        self.points_to_add = points
        self.points = []
        self.factice_points = []
        self.edges: List[Tuple[int]] = [] # Contains the indices of the points
        self.triangles: List[Tuple[int, int, int]] = [] # Contains the indices of the edges
        self.tolerance = 1e-6



    # === main function ================================================




    def __call__(self):
        """
        Compute the Delaunay triangulation of the points.
        """
        # create the initial triangulation
        self._make_container_triangle()
        for point in self.points_to_add:
            self.add_point(point)
        # remove the triangles that contain a factice point
        self.triangles = [tri for tri in self.triangles if not self._contains_factice_point(tri)]
        self._remove_factice_points()

    def add_point(self, point: tuple[float, float]):
        """
        Add a point to the triangulation.
        """
        # check that the point is not already in the triangulation
        if point in self.points:
            print(f"Point {point} is already in the triangulation")
            return
        # assert point not in self.points, "The point is already in the triangulation"
        tri = self.what_triangle(point)
        # add the point to the triangulation
        self.points.append(point)
        point_id = len(self.points) - 1
        edge = self._find_edge_if_one(tri, point)
        # check whether the tuple is empty or not
        if len(edge) == 0:
            self._add_point_in_triangle(point_id, tri)
        else:
            self._add_point_on_edge(point_id, edge, tri)
    
    def _add_point_in_triangle(
        self, 
        point_id: int, 
        tri: tuple[int, int, int]
    ) -> None:
        """
        1. First, we add the three new triangles to the triangulation.
        2. Then, we remove the triangle from the triangulation.
        3. Finally, we legalize the edges of the triangulation. 
        """
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
    
    def _add_point_on_edge(
        self, 
        point_id: int, 
        edge: tuple[int, int], 
        tri: tuple[int, int, int]
    ) -> None:
        """
        Add a point on an edge.
        """
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
        

    def _contains_factice_point(self, tri: tuple[int, int, int]) -> bool:
        """
        Check if the triangle contains a factice point.
        """
        for point_id in tri:
            if self.points[point_id] in self.factice_points:
                return True
        return False

    def _remove_factice_points(self):
        self.points = [p for p in self.points if p not in self.factice_points]
        # substract all point ids by the number of factice points in the triangles
        self.triangles = [
            (t[0] - len(self.factice_points), t[1] - len(self.factice_points), 
            t[2] - len(self.factice_points)) for t in self.triangles
        ]

    def _make_container_triangle(self):
        # create two other points such that the triangle formed by the three points
        # contains all the points of the triangulation
        x_min = min(self.points_to_add, key=lambda p: p[0])[0]
        x_max = max(self.points_to_add, key=lambda p: p[0])[0]
        y_min  = min(self.points_to_add, key=lambda p: p[1])[1]
        y_max = max(self.points_to_add, key=lambda p: p[1])[1]
        max_point = (2 * x_max, 2 * y_max)
        fict_point_1 = (x_min - 6 * (x_max - x_min), 2 * y_max)
        fict_point_2 = (2 * x_max, y_min - 6 * (y_max - y_min))
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

    def what_triangle(self, point: tuple[float, float]) -> Tuple[int, int, int]:
        """
        Find the triangle that contains the point.
        """
        min_diff = 1e6
        best_triangle = None
        for tri in self.triangles:
            is_in, diff = self._is_in_triangle(point, tri) 
            if is_in:
                return tri
            if diff < min_diff:
                min_diff = diff
                best_triangle = tri
            
        # plot very big point
        self.plot_red_triangle(best_triangle, point)
        raise ValueError("The point is not in any triangle")
    
    def _is_in_triangle(
        self, 
        point: tuple[float, float], 
        triangle: tuple[int, int, int]
    ) -> Tuple[bool, float]:
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
        diff = abs(area - (area1 + area2 + area3))
        return diff < self.tolerance, diff
    
    def _find_edge_if_one(
        self, 
        triangle: tuple[int, int, int], 
        point: tuple[float, float]
    ) -> tuple[int, int]:
        """
        Find the edge of the triangle that contains the point.
        """
        for edge in self._triangle_to_edges(triangle):
            if self._is_on_edge(point, edge):
                return edge
        return tuple()


    def _is_on_edge(self, point: tuple[float, float], edge: tuple[int, int]) -> bool:
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

    def _area(self, points: list[tuple[float, float]]) -> float:
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

    def _legalize_edge(self, point_id: int, edge: tuple[int, int]):
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

    def _flip_edge(self, point_id: int, edge: tuple[int, int], opp_point_id: int):
        """
        Flip the edge of the triangulation.
        """
        # remove the triangles that contain the edge
        self.triangles = [tri for tri in self.triangles \
            if not((edge[0] in tri) and (edge[1]  in tri))]
        self.triangles.append((point_id, edge[0], opp_point_id))
        self.triangles.append((point_id, edge[1], opp_point_id))

    def _is_legal(self, point_id: int, edge: tuple[int, int], opp_point_id: int) -> bool:
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
        assert self.points[point1] != self.points[point2], "The points \
        are at the same location."
        x1, y1 = self.points[point1][0] - self.points[sommet][0], \
            self.points[point1][1] - self.points[sommet][1]
        x2, y2 = self.points[point2][0] - self.points[sommet][0], \
            self.points[point2][1] - self.points[sommet][1]
        val = (x1 * x2 + y1 * y2) / (math.sqrt(x1**2 + y1**2) * math.sqrt(x2**2 + y2**2))
        # check if the value is between -1 and 1
        if val > 1:
            val = 1
        elif val < -1:
            val = -1
        ang = math.acos(val)
        return ang

    def _triangle_to_edges(
        self, 
        triangle: tuple[int, int, int]
    ) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
        """
        Convert a triangle to its edges.
        """
        return (
            (triangle[0], triangle[1]), 
            (triangle[1], triangle[2]), (triangle[2], triangle[0])
        )
    
    def get_edges(self) -> list[tuple[int, int]]:
        """
        Get the edges of the triangulation.
        """
        edges = []
        # TODO : I could optimize this if things get too slow.
        for triangle in self.triangles:
            edges.extend(self._triangle_to_edges(triangle))
        edges = list(set(edges))
        return edges
    
    def tri_min_angle(self, tri:Tuple[int, int, int]) -> float:
        """
        Get the minimum angle of a triangle.
        """
        angles = []
        for i in range(3):
            angles.append(self._angle(tri[i], tri[(i+1)%3], tri[(i+2)%3]))
        return min(angles)
    
    def tri_circumcenter(self, tri:Tuple[int, int, int]) -> Tuple[float, float]:
        """
        Get the circumcenter of a triangle. not the baricenter !!! 
        It can be outside the triangle.
        """
        a = self.points[tri[0]]
        b = self.points[tri[1]]
        c = self.points[tri[2]]
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





    # === plotting functions ===========================================





    def plot(self):
        """plot the points and the edges of triangles"""
        plt.scatter(*zip(*self.points))
        for tri in self.triangles:
            # plot the three edges of the triangle separately
            for edge in self._triangle_to_edges(tri):
                plt.plot(*zip(*[self.points[edge[0]], self.points[edge[1]]]), color='black')
        #plt.show()
    
    def plot_red_triangle(self, tri: tuple[int, int, int], point: tuple[float, float]):
        self.plot()
        plt.scatter(point[0], point[1], color='red', s=100)
        for i in range(3):
            plt.plot(*zip(*[self.points[tri[i]], self.points[tri[(i+1)%3]]]), color='red')
        # for edge in self._triangle_to_edges(tri):
        #     plt.plot(*zip(*[self.points[edge[0]], self.points[edge[1]]]), color='red')
    
    def plot_last3trianlges(self):
        self.plot()
        for tri in self.triangles[-3:]:
            for edge in self._triangle_to_edges(tri):
                plt.plot(*zip(*[self.points[edge[0]], self.points[edge[1]]]), color='red')

    def plot_red_edge(self, edge: tuple[int, int]):
        self.plot()
        plt.plot(*zip(*[self.points[edge[0]], self.points[edge[1]]]), color='red')
    
    def plot_red_point(self, point: tuple[float, float]):
        self.plot()
        plt.scatter(point[0], point[1], color='red', s=100)




if __name__ == "__main__":
    # create a random set of points
    points = [(random.random(), random.random()) for _ in range(100)]
    from scipy.spatial import Delaunay
    tri = Delaunay(points)
    tri  = Triangulation(points)
    tri.delaunay()
    tri.plot()