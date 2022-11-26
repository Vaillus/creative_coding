import bpy
import bmesh
import os
import sys
import numpy as np
from typing import List, Tuple
from shapely.geometry.polygon import LinearRing

dir = os.path.dirname(bpy.data.filepath)
if not dir in sys.path:
    sys.path.append(dir)
    
#import maroc_opencv
#import imp
#imp.reload(maroc_opencv)
maroc_opencv = bpy.data.texts["maroc_main"].as_module()
el = bpy.data.texts["ellipse"].as_module()
lo = bpy.data.texts["losange"].as_module()
#print_lib = bpy.data.texts["blank_script"].as_module()
#from maroc_opencv import *


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

def mid_val(a,b,multi=0.5):
    valmin = min(a,b)
    valmax = max(a,b)
    return float(valmin) + (valmax - valmin) * multi
    
def gen_middle(
        top_para:el.Arc, 
        bot_para:el.Arc, 
        top_ort:el.Arc, 
        bot_ort:el.Arc, 
        multi=0.5
    ):
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
    
    
def sort_points_clockwise(points:List[Tuple[float]]) -> List[Tuple[float]]:
    # sort the points clockwise
    # set the center as the mean of the points
    center = np.mean(points, axis=0)
    # sort the points by the angle they and the center make
    points = sorted(points, key=lambda point: np.arctan2(point[1] - center[1], point[0] - center[0]))
    return points

def mesh_outline(points: List[Tuple[float]], name:str="outline"):
    bm = bmesh.new()
    for i, pt in enumerate(points):
        if i == 0:
            bm.verts.new(pt)
        elif points[i] != points[i-1]:
            bm.verts.new(pt)
    # bmesh.ops.triangle_fill(bm, use_beauty=True, use_dissolve=False)
    #for edge in bmesh.ops.connect_verts(bm, verts=list(bm.vert):
    for i in range(len(list(bm.verts))-1):
        bm.edges.new((list(bm.verts)[i], list(bm.verts)[i+1]))
    bm.edges.new((list(bm.verts)[-1], list(bm.verts)[0]))
    new_mesh = bpy.data.meshes.new(name)
    bm.to_mesh(new_mesh)
    return new_mesh


#main = list(bpy.data.collections)[0]    
#scene = bpy.context.scene
#sw, se, nw, ne = maroc_opencv.init_borders()
#points = []
#for ellipse in  [nw, ne, se, sw]:
#    points += ellipse.get_points()
#points = sort_points_clockwise(points)
#points = np.array(points).astype("float") / 100
#points = np.append(points, np.zeros((points.shape[0], 1)), axis=1)
##points2 = points / 2
#points = points.tolist()
##points2 = points2.tolist()
#new_mesh = mesh_outline(points)
#outline = bpy.data.objects.new('losange', new_mesh)
#main.objects.link(outline)
## mettre l'outline en objet pr
##main.objects.active = outline
##bpy.context.active_object = outline

## one of the objects to join

##ctx['selected_objects'] = obs
## In Blender 2.8x this needs to be the following instead:



## creating the middle losanges
#n_mid_arcs = 4
#md = [ne]
#mg = [nw]
#offset = 0

#for i in reversed(range(0, n_mid_arcs)):
#    if offset == 0 and i == 0:
#        continue
#    mid = gen_middle(ne,sw, nw, se, (i+offset)/n_mid_arcs)
#    #mid.render(img)
#    md += [mid]
#    mid2 = gen_middle(nw, se, ne,sw, (i+offset)/n_mid_arcs)
#    #mid2.render(img)
#    mg+= [mid2]
#md += [sw]
#mg += [se]
#obs = [outline]
#for i in range(len(md)-1):
#    for j in range(len(md)-1):
#        test = lo.Losange(mg[i], md[j], md[j+1], mg[i+1], 0.15)
#        lo_points = test.get_points()
#        lo_points = sort_points_clockwise(lo_points)
#        lo_points = np.array(lo_points).astype("float") / 100
#        lo_points = np.append(lo_points, np.zeros((lo_points.shape[0], 1)), axis=1)
#        lo_points = lo_points.tolist()
#        lo_mesh = mesh_outline(lo_points)
#        
#        lo_object = bpy.data.objects.new('lo_object', lo_mesh)
#        main.objects.link(lo_object)
#        
#        obs += [lo_object]
#        #ctx['selected_editable_objects'] = lo_object

#        #bpy.ops.object.join(ctx)
#        # faudrait que je fasse un truc ici


## === join the losanges and fill the surface

#override = {}
#override['object'] = override['active_object'] = [ob for ob in bpy.context.scene.objects if ob.name.startswith("losange")][-1]
#override["selected_editable_objects"] = [ob for ob in bpy.context.scene.objects if ob.name.startswith("lo_object") or ob.name.startswith("losange")]
#override['mode'] = "OBJECT"
#with bpy.context.temp_override(**override):
#    bpy.ops.object.join()
#    me = bpy.context.object.data
#    bm = bmesh.new()
#    bm.from_mesh(me)
#    bmesh.ops.triangle_fill(
#        bm, 
#        use_beauty=True,
##        use_dissolve=True, 
#        edges=bm.edges
#    )
#    bm.to_mesh(me)
#    bm.free()

def order_transform_points(points:List):
    points = sort_points_clockwise(points)
    points = np.array(points).astype("float") / 100
    points = np.append(points, np.zeros((points.shape[0], 1)), axis=1)
    #points2 = points / 2
    points = points.tolist()
    return points

def create_outline_mesh(sw, se, nw, ne):
    points = []
    for ellipse in  [nw, ne, se, sw]:
        points += ellipse.get_points()
    points = order_transform_points(points)
    #points2 = points2.tolist()
    new_mesh = mesh_outline(points, "outline")
    
    return new_mesh

def create_object_outline(outline_mesh):
    #remove the old object if necessary and create the new one with the outline mesh
    cond = False
    for ob in list(bpy.data.objects):
        if ob.name.startswith("losange"):
            cond = True
    if cond:
        outline = bpy.data.objects["losange"]
        mesh = outline.data
        bpy.data.meshes.remove(mesh)
        #bpy.data.objects.remove(outline)
        # remove mesh from object.
        #bpy.data.meshes.remove(outline.data)
    outline = bpy.data.objects.new('losange', outline_mesh)
    main = list(bpy.data.collections)[0]
    bpy.context.collection.objects.link(outline)
    return outline
    

def create_middle_arcs(sw, se, nw, ne, offset):
    # creating the middle arcs
    n_mid_arcs = 4
    md = [ne]
    mg = [nw]
    #offset = 0

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
    
    return md, mg


def create_middle_losanges(outline, mg, md, scene):
    all_objects = [outline]
    for i in range(len(md)-1):
        for j in range(len(md)-1):
            test = lo.Losange(mg[i], md[j], md[j+1], mg[i+1], 0.15)
            points = test.get_points()
            points = order_transform_points(points)
            mesh = mesh_outline(points, "inline")
            object = bpy.data.objects.new('in_losange', mesh)
            bpy.context.collection.objects.link(object)
            all_objects += [object]      
    return all_objects

def create_final_surface(scene):
    override = {}
    override['object'] = override['active_object'] = [ob for ob in scene.objects if ob.name.startswith("losange")][-1]
    override["selected_editable_objects"] = [ob for ob in scene.objects if ob.name.startswith("in_losange") or ob.name.startswith("losange")]
    override['mode'] = "OBJECT"
    with bpy.context.temp_override(**override):
        bpy.ops.object.join()
        # il faudrait que je vire tous les meshes ?
        
        me = bpy.context.object.data
        bm = bmesh.new()
        bm.from_mesh(me)
        bmesh.ops.triangle_fill(
            bm, 
            use_beauty=True,
            edges=bm.edges
        )
        bm.to_mesh(me)
        bm.free()
    for mesh in list(bpy.data.meshes):
        if mesh.name.startswith("inline"):
            bpy.data.meshes.remove(mesh)
    #delete_all_meshes()
#    for ob in [obn for obn in bpy.context.scene.objects if obn.name.startswith("in_losange")]:
#        bpy.data.meshes.remove(ob.data)
        #bpy.data.objects.remove(ob)
    

def main_losange(scene):  
    # set offset with frame
    n_frame = float(scene.frame_current)
    period = 50.0
    offset = (n_frame % period) / period 
    
    sw, se, nw, ne = maroc_opencv.init_borders()
    outline_mesh = create_outline_mesh(sw, se, nw, ne)
    
    outline = create_object_outline(outline_mesh)
    md, mg = create_middle_arcs(sw, se, nw, ne, offset)
    all_objects = create_middle_losanges(outline, mg, md, scene)
    create_final_surface(scene)
    
    # === join the losanges and fill the surface

def delete_all_meshes():
    for mesh in list(bpy.data.meshes):
        bpy.data.meshes.remove(mesh)




main_losange(bpy.context.scene)

bpy.app.handlers.frame_change_pre.append(main_losange)