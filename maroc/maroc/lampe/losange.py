import numpy as np
from typing import Tuple, List, Generator

from maroc.lampe.arc import Arc


def interpolate(a,b,frac=0.5):
    """Return a value between two values a and b.
    frac represents the fraction of the difference between a and b.

    Args:
        a (float): the first value
        b (float): the second value
        frac (float, optional): the fraction of the difference between 
        a and b. Defaults to 0.5.

    Returns:
        float: the middle value
    """
    valmin = min(a,b)
    valmax = max(a,b)
    return float(valmin) + (valmax - valmin) * frac

class Losange:
    def __init__(
        self,
        nw:Arc,
        ne:Arc,
        sw:Arc,
        se:Arc,
        inner_arc_interpolation:bool=False,
        pad= 0.1,
        has_lines: bool=False,
        thickness: int=1
    ):
        """
        Args:
            nw (Arc): the top left arc
            ne (Arc): the top right arc
            sw (Arc): the bottom left arc
            se (Arc): the bottom right arc
            relative (bool, optional): whether the losange is relative to the 
            given arcs. Defaults to False.
            pad (float, optional): the padding between the arcs. Defaults to 0.1.
        """
        self.onw = nw
        self.one = ne
        self.osw = sw
        self.ose = se
        self.has_lines = has_lines
        self.thickness = thickness
        self.inner_arc_interpolation = inner_arc_interpolation
        self.pad = pad     
        self.inw, self.ine, self.isw, self.ise = self._generate_inner_borders()
        if self.has_lines:
            self.lines = self._generate_lines()
        else:
            self.lines = []
        # self.inner_losanges = self._generate_inner_losanges()

    def _generate_inner_borders(self):
        # generate the invisible inner arcs that delimitate the content 
        # displayed within the losange.
        # There are two ways of generating them, depending on the value
        # of the parameter `self.inner_arc_interpolation`
        if self.inner_arc_interpolation:
            inw, ine, isw, ise =\
                self._gen_inner_arcs_interpolated()
        else:
            inw, ine, isw, ise =\
                self._gen_inner_arcs_offset()
        # compute the intersection points of the inner arcs
        inw, ine, isw, ise = self._compute_inner_intersections(inw, ine, isw, ise)
        return inw, ine, isw, ise

    def _gen_inner_arcs_interpolated(self) -> Tuple[Arc, Arc, Arc, Arc]:
        """Generate the inner arcs with an interpolation between the 
        outer arcs."""
        inw = Losange.gen_inner_arc_interpolated(self.onw, self.ose, \
            self.one, self.osw, 
            1.0 - self.pad)
        ine = Losange.gen_inner_arc_interpolated(self.one, self.osw, \
            self.onw, self.ose, 
            1.0 - self.pad)
        isw = Losange.gen_inner_arc_interpolated(self.osw, self.one, \
            self.onw, self.ose, 
            self.pad)
        ise = Losange.gen_inner_arc_interpolated(self.ose, self.onw, \
            self.osw, self.one, 
            self.pad)
        return inw, ine, isw, ise
        
    @staticmethod
    def gen_inner_arc_interpolated(
        para1:Arc, 
        para2:Arc, 
        orth1:Arc, 
        orth2:Arc, 
        frac=0.5
    ) -> Arc:
        """Generate an arc of position and proportions are an 
        interpolation between two given (parallel) arcs, and whose 
        limits are defined by two given orthogonal arcs."""
        # TODO : i compute the min and the max of all the values separately but I should probably not do that
        center_x = interpolate(para1.center[0], para2.center[0], frac)
        center_y = interpolate(para1.center[1], para2.center[1], frac)
        center = (center_x, center_y)
        a = interpolate(para1.a, para2.a, frac)
        b = interpolate(para1.b, para2.b, frac)
        arc = Arc(center, a=a, b=b)
        top_x, top_y = arc.intersect(orth1)
        arc.tp = (top_x, top_y)
        bot_x, bot_y = arc.intersect(orth2)
        arc.bp = (bot_x, bot_y)
        return arc
        
    def _gen_inner_arcs_offset(self) -> Tuple[Arc, Arc, Arc, Arc]:
        """Generate the inner arcs with a specified relative position to 
        the outer arcs."""
        inw = Losange._gen_inner_arc_offset(self.onw, [1, -1])
        ine = Losange._gen_inner_arc_offset(self.one, [-1, -1])
        isw = Losange._gen_inner_arc_offset(self.osw, [1, 1])
        ise = Losange._gen_inner_arc_offset(self.ose, [-1, 1])
        return inw, ine, isw, ise

    @staticmethod
    def _gen_inner_arc_offset(arc: Arc, rel_pos:List[int]) -> Arc:
        """Generate an arc with the same proportions as the arc passed 
        as argument but with a specified relative position to the 
        given arc"""
        new_arc = Arc(
            tuple(np.array(arc.center) + np.array(rel_pos)),
            top_point=tuple(np.array(arc.tp) + np.array(rel_pos)),
            bottom_point=tuple(np.array(arc.bp) + np.array(rel_pos))
        )
        return new_arc
    
    def _compute_inner_intersections(
        self, inw:Arc, ine:Arc, isw:Arc, ise:Arc
    ) -> Tuple[Arc, Arc, Arc, Arc]:
        """Compute the intersection points of the inner arcs."""
        inw.tp = inw.intersect(ine)
        ine.tp = inw.tp
        inw.bp = inw.intersect(isw)
        isw.tp = inw.bp
        ine.bp = ine.intersect(ise)
        ise.tp = ine.bp
        isw.bp = isw.intersect(ise)
        ise.bp = isw.bp
        return inw, ine, isw, ise
    
    def _generate_lines(self):
        """Generate three arcs equally spaced between inw and ise"""
        lines = []
        n_lines = 3
        for i in range(n_lines-1):
            lines.append(Losange.gen_inner_arc_interpolated(
                self.inw, self.ise, self.ine, self.isw, 
                float(i+1)/n_lines
            ))
        return lines

    def _generate_inner_losanges(self) -> List:
        if self.lines == []:
            return []
        clines = [self.inw]
        for line in self.lines[::-1]:
            clines.append(line)
        clines += [self.ise]
        inner_losanges = []
        for i in range(len(clines)-1):
            inner_losanges.append(Losange(
                clines[i],
                self.ine,
                self.isw,
                clines[i+1],
                inner_arc_interpolation=True,
                pad = 0.1,
                has_lines=False
            ))
        return inner_losanges
        
    
        
    # === plotting / point accessing ===================================



    def render(
        self, 
        img: np.ndarray[int, np.dtype[np.int64]], 
        color: Tuple[int]=(0,0,0),
        max_x:int=0,
        min_x:int=0,
        max_y:int=0,
        min_y:int=0
    ) -> None:
        bold = self.thickness == 2
        self.inw.render(img, color, bold=bold, max_x=max_x, min_x=min_x, max_y=max_y, min_y=min_y)
        self.ine.render(img, color, bold=bold, max_x=max_x, min_x=min_x, max_y=max_y, min_y=min_y)
        self.isw.render(img, color, bold=bold, max_x=max_x, min_x=min_x, max_y=max_y, min_y=min_y)
        self.ise.render(img, color, bold=bold, max_x=max_x, min_x=min_x, max_y=max_y, min_y=min_y)  
        if self.has_lines:
            for line in self.lines:
                line.render(img, color, bold=bold)
    
