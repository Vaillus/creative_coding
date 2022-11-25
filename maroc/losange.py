import ellipse as el

import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry.polygon import LinearRing
from typing import Tuple, List

def mid_val(a,b,multi=0.5):
    valmin = min(a,b)
    valmax = max(a,b)
    return float(valmin) + (valmax - valmin) * multi

# === ellipses intersection ============================================


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
        self.lines = None
        self.thickness = thickness
        
        self.inw = None
        self.ine = None
        self.isw = None
        self.ise = None
        self.generate_inner_borders()
        self.generate_lines()
    
    def gen_relative_arc(self, arc: el.Arc, rel_pos:List[int]) -> el.Arc:

        new_arc = el.Arc(
            tuple(np.array(arc.center) + np.array(rel_pos)),
            tuple(np.array(arc.tp) + np.array(rel_pos)),
            tuple(np.array(arc.bp) + np.array(rel_pos))
        )
        return new_arc

    def generate_inner_borders(self):
        # genrate the ovals that are inside the losange
        # copy onw into self.inw
        if self.relative:
            self.inw = Losange.gen_middle(self.onw, self.ose, self.one, self.osw, 1.0 - self.pad)
            self.ine = Losange.gen_middle(self.one, self.osw, self.onw, self.ose, 1.0 - self.pad)
            self.isw = Losange.gen_middle(self.osw, self.one, self.ose, self.onw, self.pad)
            self.ise = Losange.gen_middle(self.ose, self.onw, self.osw, self.one, self.pad)
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

    def generate_lines(self):
        """Generate three arcs equally spaced between inw and ise"""
        if self.has_lines:
            self.lines = []
            self.lines.append(Losange.gen_middle(self.inw, self.ise, self.ine, self.isw, 1.0/4))
            self.lines.append(Losange.gen_middle(self.inw, self.ise, self.ine, self.isw, 2.0/4))
            self.lines.append(Losange.gen_middle(self.inw, self.ise, self.ine, self.isw, 3.0/4))

    @staticmethod
    def gen_middle(para1:el.Arc, para2:el.Arc, orth1:el.Arc, orth2:el.Arc, multi=0.5) -> el.Oval:
        # TODO : i compute the min and the max of all the values separately but I should probably not do that
        x = mid_val(para1.center[0], para2.center[0], multi)
        y = mid_val(para1.center[1], para2.center[1], multi)
        center = (x,y)
        a = mid_val(para1.a, para2.a, multi)
        b = mid_val(para1.b, para2.b, multi)
        new_el = el.Oval(center, a, b)
        x, y = new_el.intersect(orth1)
        new_el.tp = (x,y)
        x, y = new_el.intersect(orth2)
        new_el.bp = (x,y)
        return new_el

    def render(self, img, color=(0,0,0)):
        bold = self.thickness == 2
        self.inw.render(img, color, bold=bold)
        self.ine.render(img, color, bold=bold)
        self.isw.render(img, color, bold=bold)
        self.ise.render(img, color, bold=bold)  
        if self.has_lines:
            for line in self.lines:
                line.render(img, color, bold=bold)

