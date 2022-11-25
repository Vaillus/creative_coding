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
        pad= 0.1
    ):
        self.onw = nw
        self.one = ne
        self.osw = sw
        self.ose = se
        self.pad = pad
        
        self.inw = None
        self.ine = None
        self.isw = None
        self.ise = None
        self.generate_inner_borders()
    
    def generate_inner_borders(self):
        # genrate the ovals that are inside the losange
        self.inw = Losange.gen_middle(self.onw, self.ose, self.one, self.osw, 1.0 - self.pad)
        self.ine = Losange.gen_middle(self.one, self.osw, self.onw, self.ose, 1.0 - self.pad)
        self.isw = Losange.gen_middle(self.osw, self.one, self.ose, self.onw, self.pad)
        self.ise = Losange.gen_middle(self.ose, self.onw, self.osw, self.one, self.pad)
        # compute the intersection points between the inner ovals
        self.inw.tp = self.inw.intersect(self.ine)
        self.inw.bp = self.inw.intersect(self.isw)
        self.ine.tp = self.ine.intersect(self.inw)
        self.ine.bp = self.ine.intersect(self.ise)
        self.isw.tp = self.isw.intersect(self.inw)
        self.isw.bp = self.isw.intersect(self.ise)
        self.ise.tp = self.ise.intersect(self.ine)
        self.ise.bp = self.ise.intersect(self.isw)

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
        self.inw.render(img, color)
        self.ine.render(img, color)
        self.isw.render(img, color)
        self.ise.render(img, color)  

    def 

