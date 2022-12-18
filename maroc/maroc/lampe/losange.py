import numpy as np
from typing import Tuple, List, Generator

from maroc.lampe import ellipse as el


def mid_val(a,b,multi=0.5):
    valmin = min(a,b)
    valmax = max(a,b)
    return float(valmin) + (valmax - valmin) * multi

class Losange:
    def __init__(
        self,
        nw:el.Arc,
        ne:el.Arc,
        sw:el.Arc,
        se:el.Arc,
        relative:bool=False,
        pad= 0.1,
        has_lines: bool=False,
        thickness: int=1
    ):
        self.onw = nw
        self.one = ne
        self.osw = sw
        self.ose = se
        self.relative = relative
        self.pad = pad
        self.has_lines = has_lines
        self.lines = []
        self.inner_losanges = []
        self.thickness = thickness
        
        self.inw = None
        self.ine = None
        self.isw = None
        self.ise = None
        self._generate_inner_borders()
        self._generate_lines()
        self._generate_inner_losanges()

    def _generate_inner_borders(self):
        # generate the ovals that are inside the losange
        # copy onw into self.inw
        if self.relative:
            self.inw = Losange.gen_middle(self.onw, self.ose, self.one, \
                self.osw, 1.0 - self.pad)
            self.ine = Losange.gen_middle(self.one, self.osw, self.onw, \
                self.ose, 1.0 - self.pad)
            self.isw = Losange.gen_middle(self.osw, self.one, self.ose, \
                self.onw, self.pad)
            self.ise = Losange.gen_middle(self.ose, self.onw, self.osw, \
                self.one, self.pad)
        else:
            self.inw = self.gen_relative_arc(self.onw, [1, -1])
            self.ine = self.gen_relative_arc(self.one, [-1, -1])
            self.isw = self.gen_relative_arc(self.osw, [1, 1])
            self.ise = self.gen_relative_arc(self.ose, [-1, 1])
        # compute the intersection points between the inner ovals
        self.inw.tp = self.inw.intersect(self.ine)
        self.inw.bp = self.inw.intersect(self.isw)
        self.ine.tp = self.ine.intersect(self.inw)
        self.ine.bp = self.ine.intersect(self.ise)
        self.isw.tp = self.isw.intersect(self.inw)
        self.isw.bp = self.isw.intersect(self.ise)
        self.ise.tp = self.ise.intersect(self.ine)
        self.ise.bp = self.ise.intersect(self.isw)
    
    def _generate_lines(self):
        """Generate three arcs equally spaced between inw and ise"""
        if self.has_lines:
            self.lines = []
            n_lines = 3
            for i in range(n_lines-1):
                self.lines.append(Losange.gen_middle(
                    self.inw, self.ise, self.ine, self.isw, 
                    float(i+1)/n_lines
                ))
            # self.lines.append(Losange.gen_middle(self.inw, self.ise, self.ine, \
            #     self.isw, 2.0/4))
            # self.lines.append(Losange.gen_middle(self.inw, self.ise, self.ine, \
            #     self.isw, 3.0/4))

    @staticmethod
    def gen_middle(
        para1:el.Arc, 
        para2:el.Arc, 
        orth1:el.Arc, 
        orth2:el.Arc, 
        multi=0.5
    ) -> el.Arc:
        """Generate the middle arc between two parallel arcs and two
        orthogonal arcs"""
        # TODO : i compute the min and the max of all the values separately but I should probably not do that
        x = mid_val(para1.center[0], para2.center[0], multi)
        y = mid_val(para1.center[1], para2.center[1], multi)
        center = (x,y)
        a = mid_val(para1.a, para2.a, multi)
        b = mid_val(para1.b, para2.b, multi)
        new_el = el.Arc(center, a=a, b=b)
        x, y = new_el.intersect(orth1)
        new_el.tp = (x,y)
        x, y = new_el.intersect(orth2)
        new_el.bp = (x,y)
        return new_el
    
    def gen_relative_arc(self, arc: el.Arc, rel_pos:List[int]) -> el.Arc:
        """Generate an arc with specified relative position to the 
        given arc"""
        new_arc = el.Arc(
            tuple(np.array(arc.center) + np.array(rel_pos)),
            top_point=tuple(np.array(arc.tp) + np.array(rel_pos)),
            bottom_point=tuple(np.array(arc.bp) + np.array(rel_pos))
        )
        return new_arc
    
    def _generate_inner_losanges(self):
        if self.lines == []:
            return
        clines: List[el.Arc] = [self.inw]
        for line in self.lines[::-1]:
            clines.append(line)
        clines += [self.ise]
        for i in range(len(clines)-1):
            self.inner_losanges.append(Losange(
                clines[i],
                self.ine,
                self.isw,
                clines[i+1],
                relative=True,
                pad = 0.1,
                has_lines=False
            ))
        
    
        




    # === plotting / point accessing ===================================





    def render(self, img, color=(0,0,0)):
        bold = self.thickness == 2
        self.inw.render(img, color, bold=bold)
        self.ine.render(img, color, bold=bold)
        self.isw.render(img, color, bold=bold)
        self.ise.render(img, color, bold=bold)  
        if self.has_lines:
            for line in self.lines:
                line.render(img, color, bold=bold)
    
    def get_points(self, n_points:int=10, outer:bool=False) -> List[Tuple[float, float]]:
        """Returns the points of the losange"""
        # TODO : handle the case with the middle lines
        points = []
        for id, arc in enumerate(self.get_outline(outer=outer)):
            # the arcs are always called in the same order : nw, ne, se, sw
            # and we know that the first two arcs must generate the points 
            # clockwise and the last two anticlockwise.
            if id == 0 or id == 1:
                clockwise = True
            else:
                clockwise = False
            arc_points = arc.get_points(n_points=n_points, clockwise=clockwise)
            if points == []:
                points = arc_points
            else:
                if arc_points[0] == points[-1]:
                    points.extend(arc_points[1:])
                else:
                    points.extend(arc_points)
        if points[0] == points[-1]:
            points.pop()
        #points = list(set(points))
        return points

    def get_outline(self, outer:bool=False) -> Generator[el.Arc, None, None]:
        """Genreator for getting the four arcs of the losange"""
        if outer:
            for arc in [self.onw, self.one, self.ose, self.osw]:
                yield arc
        else:
            for arc in [self.inw, self.ine, self.ise, self.isw]:
                yield arc
    
    def get_points_edges(
        self, 
        n_points:int=10,
        outer:bool=False,
    ) -> Tuple[List[Tuple[float, float]], List[Tuple[int, int]]]:
        if self.has_lines:
            points, edges = [], []
            for losange in self.inner_losanges:
                lo_points, lo_edges = losange.get_points_edges(n_points=n_points, outer=outer)
                lo_edges = [(p[0]+len(points), p[1]+len(points)) for p in lo_edges]
                points += lo_points
                edges += lo_edges
        else:
            points = self.get_points(n_points=n_points, outer=outer)
            edges = []
            for i in range(len(points)):
                edges.append((i, (i+1)%len(points)))
        return points, edges

def sort_points_clockwise(points:List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    # sort the points clockwise
    # set the center as the mean of the points
    center = np.mean(points, axis=0)
    # sort the points by the angle they and the center make
    points = sorted(points, key=lambda point: np.arctan2(point[1] - center[1], point[0] - center[0]))
    return points

