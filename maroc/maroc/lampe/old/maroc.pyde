import math
def setup():
    size(400,400)
    background(255)

def draw():
    print("begin")
    center = 200
    wid = 200
    hei = 300
    b_border = 350
    print("yo")
    base = (center,b_border)
    left = (center-int(wid/2), b_border -100)
    right = (center+int(wid/2), b_border -100)
    top = (center, b_border - hei)
    print("top: "+str(top))
    lcenter = (100, 390)
    rcenter = (300, 390)
    print("yo")
    #line(200,0, 200, 400)
    #point(*lcenter)
    #draw_ellipse((200,200), 200, 100)
    #full_draw_arc((200,200), (201,301), (298, 198))
    sw = init_arc(lcenter, left, base)
    sw.display() # ! purposefully not available anymore
    #sw.convert_point_rad(left)
    se = init_arc(rcenter, right, base)
    se.display()
    nw = init_arc(rcenter, top, left)
    nw.display()
    ne = init_arc(lcenter, top, right)
    ne.display()
    compute_ellipses_intersections(sw, se)
    
    display_middle(ne,sw, nw, sw)
    
def init_arc(center, top_point, bottom_point):
    a,b = compute_ellipse_from_three(center, top_point, bottom_point)
    seg = Segment(center, a, b, top_point, bottom_point)
    print("hehehe")
    return seg

def display_middle(top_para, bot_para, top_ort, bot_ort):
    multi = 0.5
    #print(str(arc1.center[0]))
    center = (float(top_para.center[0] + bot_para.center[0]) * multi, float(top_para.center[1] + bot_para.center[1]) * multi)
    #tang = float(arc1.tang + arc2.tang) * multi
    #bang = float(arc1.bang + arc2.bang) * multi
    a = float(top_para.a + bot_para.a) * multi
    b = float(top_para.b + bot_para.b) * multi
    
    tp = (float(top_para.tp[0] + bot_para.tp[0]) * multi, float(top_para.tp[1] + bot_para.tp[1]) * multi)
    bp = (float(top_para.bp[0] + bot_para.bp[0]) * multi, float(top_para.bp[1] + bot_para.bp[1]) * multi)
    new_arc = Segment(center, a, b, tp, bp)
    new_arc.display()
    

def full_draw_arc(center, p1, p2):
    #print("center: "+str(center))
    #print("point 1: "+str(p1))
    #print("point 2:"+str(p2))
    a,b = compute_ellipse_from_three(center, p1, p2)
    #print(a)
    #print(b)
    #draw_ellipse(center, a, b)
    draw_arc(center, a,b, p1, p2)

def compute_ellipse_from_three(center, p1, p2):
    #print("compute a and b")
    Ax = (p1[0]- center[0]) ** 2
    Ay = (p1[1]- center[1]) ** 2
    Bx = (p2[0]- center[0]) ** 2
    By = (p2[1]- center[1]) ** 2
    #eps = 1
    num = Ay - By
    den = Bx - Ax
    # handle exceptions
    o = float(num)/float(den)
    #print("Ax:"+str(Ax))
    #print("Ay:"+str(Ay))
    #print("Bx:"+str(Bx))
    #print("By:"+str(By))
    #print("o:"+str(o))
    #print(Ax - Ay/o)
    a = sqrt(float(Ax) + float(Ay)/o)
    #print("s")
    b = sqrt(float(Ax)*o + float(Ay))
    #print("a: "+str(a))
    #print("b: "+str(b))
    
    return a, b

def draw_arc(center, a,b, p1, p2):
    xmax = max(p1[0],p2[0])
    xmin = min(p1[0],p2[0])
    ymax = max(p1[1],p2[1])
    ymin = min(p1[1],p2[1])
    for x in range(int(-a), int(a)):
        if (x+center[0] <= xmax) and (x+center[0]>=xmin):
            yp = b * sqrt(1 - (x/a)**2)
            ym = - yp
            if (yp+center[1] <= ymax) and (yp+center[1]>=ymin):
                point(x+center[0], yp+center[1])
            if (ym+center[1] <= ymax) and (ym+center[1]>=ymin):
                point(x+center[0],ym+center[1])
    



def draw_ellipse(center, a, b):
    for x in range(int(-a), int(a)):
        x = float(x)
        #x/=res
        yp = b * sqrt(1 - (x/a)**2)
        #print("===== before scaling")
        #print(x)
        #print(yp)
        #yp*=b
        ym = - yp
        #x*=a
        #print("=== after scaling")
        #print(x)
        #print(yp)
        #point(x+center[0],10)
        point(x+center[0], yp+center[1])
        point(x+center[0],ym+center[1])
    #print("=============================")
    
    
class Segment():
    def __init__(self, center, a, b, top_point=None, bottom_point=None):
        self.center = center
        self.a = a
        self.b = b
        self.tp = top_point
        self.bp = bottom_point
        #self.tang = self.convert_point_rad(self.tp)
        #self.bang = self.convert_point_rad(self.bp)
        
    def display(self):
        draw_arc(self.center, self.a, self.b, self.tp, self.bp)
    
    def convert_point_rad(self, pt):
        res = atan2(pt[1] - self.center[1], pt[0] - self.center[0])
        #print("relative x axis : "+str(pt[0] - self.center[0]))
        #print("relative y axis : "+str(pt[1] - self.center[1]))
        #print("angle: "+str(res))
        return res

def compute_ellipses_intersections(ell1, ell2):
    # small variables
    a = ell1.center[0]
    b = ell1.a ** 2
    c = ell1.center[1]
    k = ell1.b ** 2
    e = ell2.center[0]
    f = ell2.a ** 2
    g = ell2.center[1]
    h = ell2.b ** 2
    # big variables
    A = 1/b - 1/f
    B = -2*a/b + 2*e/f
    C = 1/k - 1/h
    D = -2*c/k + 2*g/h
    E = a**2/b + c**2/k + e**2/f + g**2/h
    # compute the intersection points of the ellipses
    # with the big variables
    # x1, x2 = solve_quadratic(A, B, E)
    # y1, y2 = solve_quadratic(C, D, E)
    # print("x1: "+str(x1))
    # print("x2: "+str(x2))
    # print("y1: "+str(y1))
    # print("y2: "+str(y2))
    x = 
    y0, y1 = compute_y(x, A, B, D, E)

def solve_quadratic(A,B,E):
    print("solve quadratic")
    print(A)
    print(B)
    print(E)
    print("=====")
    print(B**2 - 4*A*E)
    if B**2 - 4*A*E < 0:
        return None
    x1 = (-B + sqrt(B**2 - 4*A*E)) / (2*A)
    x2 = (-B - sqrt(B**2 - 4*A*E)) / (2*A)
    return x1, x2

def compute_y(x, A, B, C, D, E):
    delta = D**2 - 4*(A*x**2 +B*x + E) * C
    if delta < 0:
        return None
    y1 = (-D + sqrt(delta)) / (2*C)
    y2 = (-D - sqrt(delta)) / (2*C)
    return y1, y2
