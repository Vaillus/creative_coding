import losange as lo
import ellipse as el
from delaunay import Triangulation
from typing import List, Tuple, Optional
from shapely.geometry.polygon import LinearRing
import numpy as np

class Schema:
    def __init__(self):
        self.points : List[float] = []
        self.triangles : List[int]

    def get_list_points(self):
        sw, se, nw, ne = self._init_outline()
        n_mid_arcs = 4
        md: List[el.Arc] = [ne]
        mg = [nw]
        offset = 0
        for i in reversed(range(0, n_mid_arcs)):
            if offset == 0 and i == 0:
                continue
            mid = self._gen_middle(ne,sw, nw, se, (i+offset)/n_mid_arcs)
            #mid.render(img)
            md += [mid]
            mid2 = self._gen_middle(nw, se, ne,sw, (i+offset)/n_mid_arcs)
            #mid2.render(img)
            mg+= [mid2]
        md += [sw]
        mg += [se]





    # === main parts creation  =========================================





    def _init_outline(self) -> Tuple[el.Arc]:
        # position variables
        # TODO : put the variables in th class variables
        center = 200
        wid = 200
        hei = 300
        b_border = 350
        base = (b_border, center)
        left = (b_border -100, center-int(wid/2))
        right = (b_border -100, center+int(wid/2))
        top = (b_border - hei, center)
        lcenter = (390, 100)
        rcenter = (390, 300)
        # initialize borders
        sw = el.Arc(lcenter, left, base)
        se = el.Arc(rcenter, right, base)
        nw = el.Arc(rcenter, top, left)
        ne = el.Arc(lcenter, top, right)
        return sw, se, nw, ne

    def _gen_middle(
        self, 
        top_para:el.Arc, 
        bot_para:el.Arc, 
        top_ort:el.Arc, 
        bot_ort:el.Arc, 
        multi:float=0.5
    ) -> el.Oval:
        x = self._mid_val(top_para.center[0], bot_para.center[0], multi)
        y = self._mid_val(top_para.center[1], bot_para.center[1], multi)
        center = (x,y)
        a = self._mid_val(top_para.a, bot_para.a, multi)
        b = self._mid_val(top_para.b, bot_para.b, multi)
        new_ellipse = el.Oval(center, a, b)
        x, y = self._find_ellipses_intersection(new_ellipse, top_ort)
        new_ellipse.tp = (x,y)
        x, y = self._find_ellipses_intersection(new_ellipse, bot_ort)
        new_ellipse.bp = (x,y)
        return new_ellipse
    
    def _mid_val(self, a, b, multi=0.5) -> float:
        valmin = min(a,b)
        valmax = max(a,b)
        return float(valmin) + (valmax - valmin) * multi

    def _find_ellipses_intersection(
        self, 
        ell1:el.Oval, 
        ell2:el.Oval
    ) -> Tuple[float]:
        # convert the ellipses into a list of points
        a, b = self._ellipse_polyline(
            [(ell1.center[0], ell1.center[1], ell1.a, ell1.b, 0), 
            (ell2.center[0], ell2.center[1], ell2.a, ell2.b, 0)]
        )
        # find the intersection points between the two ellipses
        x, y = self._intersections(a, b)
        # find the point that is between the two limit points of ell2
        x, y = self._sel_good_point(x, y, ell2)
        return x, y


    # ?


    def _sel_good_point(
        self, 
        x:List[float], 
        y:List[float], 
        ell2:el.Oval
    ) -> Tuple[float]:
        """find the point that is between the two limit points of ell2
        """
        xmin = min(ell2.tp[0], ell2.bp[0])
        xmax = max(ell2.tp[0], ell2.bp[0])
        ymin = min(ell2.tp[1], ell2.bp[1])
        ymax = max(ell2.tp[1], ell2.bp[1])
        for i in range(len(x)):
            xi = x[i]
            yi = y[i]
            if xi <= xmax and xi >= xmin and yi <= ymax and yi >= ymin:
                x = xi
                y = yi
                break
        # assert type of x is float
        assert type(x) is float, "x is not a float"
        return x, y

    def _ellipse_polyline(self, ellipses, n=100):
        t = np.linspace(0, 2*np.pi, n, endpoint=False)
        st = np.sin(t)
        ct = np.cos(t)
        result = []
        for x0, y0, a, b, angle in ellipses:
            angle = np.deg2rad(angle)
            sa = np.sin(angle)
            ca = np.cos(angle)
            p = np.empty((n, 2))
            p[:, 0] = x0 + a * ca * ct - b * sa * st
            p[:, 1] = y0 + a * sa * ct + b * ca * st
            result.append(p)
        return result

    def intersections(self,a, b):
        ea = LinearRing(a)
        eb = LinearRing(b)
        mp = ea.intersection(eb)
        try:
            x = [p.x for p in mp]
            y = [p.y for p in mp]
        except:
            print("no intersection")
            plt.plot(a[:,0], a[:,1])
            plt.plot(b[:,0], b[:,1])
        return x, y
