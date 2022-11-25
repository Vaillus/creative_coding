import cv2
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry.polygon import LinearRing
from typing import Tuple, List

import ellipse as el
import losange as lo
from PIL import Image

def test():
    img = np.zeros((400, 400, 3), dtype = "uint8")
    
    cv2.rectangle(img, (0, 0), (250, 250), (255, 255, 255), -1)
    #draw a red line
    cv2.line(img, (0, 0), (250, 250), (0, 0, 255), 1 )
    cv2.imshow("output", img)
    cv2.waitKey(0) 

def sort_points_clockwise(points:List[Tuple[float]]) -> List[Tuple[float]]:
    # sort the points clockwise
    # set the center as the mean of the points
    center = np.mean(points, axis=0)
    # sort the points by the angle they and the center make
    points = sorted(points, key=lambda point: np.arctan2(point[1] - center[1], point[0] - center[0]))
    return points

def base():
    # initialize image
    
    sw, se, nw, ne = init_borders()
    T = 50
    t = 0
    imgs = []
    while t < T:
        if t == T:
            t=0
        offset = t/T
        t += 1
        img = np.ones((400, 400, 3), dtype = "uint8") * 255
        sw.render(img)
        se.render(img)
        nw.render(img)
        ne.render(img)
        # initialize middle arcs
        n_mid_arcs = 4
        md = [ne]
        mg = [nw]

        for i in reversed(range(0, n_mid_arcs)):
            if offset == 0 and i == 0:
                continue
            mid = gen_middle(ne,sw, nw, se, (i+offset)/n_mid_arcs)
            #mid.render(img)
            md += [mid]
            mid2 = gen_middle(nw, se, ne,sw, (i+offset)/n_mid_arcs)
            #mid2.render(img)
            mg+= [mid2]
        md += [sw]
        mg += [se]
        # display image
        col = (255,255,0)
        #mg[0].render(img, color=col)
        #mg[1].render(img, color=col)
        #md[0].render(img, color=col)
        #md[1].render(img, color=col)
        # test = lo.Losange(mg[0], md[1], md[2], mg[1])
        # test.render(img)
        for i in range(len(md)-1):
            for j in range(len(md)-1):
                test = lo.Losange(mg[i], md[j], md[j+1], mg[i+1], 0.15)
                test.render(img)
        imgs += [Image.fromarray(img)]
        cv2.imshow("output", img)
        # cv2.waitKey(0)
        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            break
    imgs[0].save('test.gif', save_all=True, append_images=imgs[1:], optimize=False, duration=50, loop=0)

def init_borders() -> Tuple[el.Arc]:
    # position variables
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
    
def gen_middle(top_para:el.Arc, bot_para:el.Arc, top_ort:el.Arc, bot_ort:el.Arc, multi=0.5):
    x = mid_val(top_para.center[0], bot_para.center[0], multi)
    y = mid_val(top_para.center[1], bot_para.center[1], multi)
    center = (x,y)
    a = mid_val(top_para.a, bot_para.a, multi)
    b = mid_val(top_para.b, bot_para.b, multi)
    new_ellipse = el.Oval(center, a, b)
    x, y = find_ellipses_intersection(new_ellipse, top_ort)
    new_ellipse.tp = (x,y)
    x, y = find_ellipses_intersection(new_ellipse, bot_ort)
    new_ellipse.bp = (x,y)
    return new_ellipse

def mid_val(a,b,multi=0.5):
    valmin = min(a,b)
    valmax = max(a,b)
    return float(valmin) + (valmax - valmin) * multi





# === ellipses intersection ============================================





def find_ellipses_intersection(ell1:el.Oval, ell2:el.Oval) -> Tuple[float]:
    # convert the ellipses into a list of points
    a, b = ellipse_polyline(
        [(ell1.center[0], ell1.center[1], ell1.a, ell1.b, 0), 
        (ell2.center[0], ell2.center[1], ell2.a, ell2.b, 0)]
    )
    # find the intersection points between the two ellipses
    x, y = intersections(a, b)
    # find the point that is between the two limit points of ell2
    x, y = sel_good_point(x, y, ell2)
    return x, y

def sel_good_point(x:List[float], y:List[float], ell2:el.Oval) -> Tuple[float]:
    """find the point that is between the two limit points of ell2

    Args:
        x (List[float]): 
        y (List[float]): 
        ell2 (el.Oval):

    Returns:
        Tuple[float]: 
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

def ellipse_polyline(ellipses, n=100):
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

def intersections(a, b):
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




if __name__ == "__main__":
    #test()
    base()