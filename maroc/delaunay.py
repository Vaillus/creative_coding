from typing import List, Tuple
import matplotlib.pyplot as plt
import random

class Triangulation:
    def __init__(self, points: List[Tuple[float]]):
        self.points = points
        self.edges: List[Tuple[int]] = [] # Contains the indices of the points
        self.triangles: List[Tuple[int]] = [] # Contains the indices of the edges
    
    def plot(self):
        """plot the points and the edges"""
        plt.scatter(*zip(*self.points))
        for edge in self.edges:
            plt.plot(*zip(*[self.points[i] for i in edge]))
        plt.show()

    def _legalize_edge(self, point: tuple[float], edge: tuple[int]):
        """
        Legalize the edge of the triangulation.
        """
        pass

    def _max_point(self):
        """
        Find the lexicographically highest point.
        1. Find the set of points with the highest y-coordinate.
        2. Among these points, find the one with the highest x-coordinate.
        """
        # Find the set of points with the highest y-coordinate
        max_y = max(self.points, key=lambda p: p[1])[1]
        max_points = [p for p in self.points if p[1] == max_y]
        # Among these points, find the one with the highest x-coordinate
        max_x = max(max_points, key=lambda p: p[0])[0]
        max_point = [p for p in max_points if p[0] == max_x][0]
        return max_point

    def _make_container_triangle(self) -> Triangulation:
        # get the point with the highest y and x coordinates
        # max_point = self._max_point()
        # create two other points such that the triangle formed by the three points
        # contains all the points of the triangulation
        x_min, x_max = min(self.points, key=lambda p: p[0])[0], max(self.points, key=lambda p: p[0])[0]
        y_min, y_max = min(self.points, key=lambda p: p[1])[1], max(self.points, key=lambda p: p[1])[1]
        # the container triangle is a triangle with vertices 
        # (x_min - 2 * (xmax - xmin), y_max), 
        # (x_max, y_min - 2 * (ymax - ymin)), 
        # (x_max, y_max)
        container_triangle = [
            (x_min - 2 * (x_max - x_min), y_max), 
            (x_max, y_min - 2 * (y_max - y_min)), 
            (x_max, y_max)
        ]
        init_trln = Triangulation(container_triangle)
        init_trln.edges = [(0, 1), (1, 2), (2, 0)]
        init_trln.triangles = [(0, 1, 2)]
        return init_trln

    def find_triangle(self, point: tuple[float]) -> int:
        """
        Find the triangle that contains the point.
        """
        

    def delaunay(self):
        """
        Compute the Delaunay triangulation of the points.
        """
        # create the initial triangulation
        init_point = self._max_point()
        init_trln = self._make_container_triangle()
        for point in self.points:
            if point != init_point:
                pass

class ContainerTriang(Triangulation):
    def __init__(self, points: list[tuple[float]]):
        super().__init__(points)

    


        
        
        



def legal_triangulation(
    points:list[tuple[float]], 
    edges:list[tuple[int]]
) -> list[tuple[int]]:
    """
    Return a legal triangulation of the points, given the edges.
    """
    pass




if __name__ == "__main__":
    # create a random set of points
    points = [(random.random(), random.random()) for _ in range(10)]
    tri  = Triangulation(points)
    tri.edges = [(0,1), (1,2), (2,0)]
    tri.plot()