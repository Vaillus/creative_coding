from typing import Tuple

import numpy as np
import cv2
import maroc.toolkit.toolkit as tk

class HexTexture:
    def __init__(self, height: float, width: float, scale:float=1.0):
        self.hei = height
        self.wid = width
        self.scale = scale
        n_row = int(10 / scale)
        n_col = int(10 / scale)
        self.pos = self._make_reg_pos(n_row, n_col)
        self.texture = np.ones((height+1, width+1, 3), dtype=np.uint8) * 255
        self.render_triangles()

    def _make_reg_pos(self, n_row:int, n_col:int):
        # Initialize coordinates on a grid from 0 to n_row - 1 
        # (9 in the test example)
        x_axis = np.array([[i for i in reversed(range(n_col))]] * (n_row+1))
        y_axis = np.array([[i for i in reversed(range(n_row+1))]] * n_col).T
        # concatenate the x and y axes
        pos = np.concatenate((
                np.expand_dims(x_axis, axis=0), 
                np.expand_dims(y_axis, axis=0)
            ), axis=0
        )
        # transform the points coordinates to have a hexagonal distribution
        pos = pos.astype(np.float64)
        # pos[1] *= np.sqrt(3)/2
        pos[0,1::2] += 0.5
        pos[0] *= 2/np.sqrt(3)
        # scale the coordinates to the size of the texture
        pos[0] *= (self.hei-1) / n_col
        pos[1] *= (self.wid-1) / n_row
        # pos is of shape (2, n_row, n_col)
        # filter out the points that are outside the texture
        # mask = (pos[0] >= 0) & (pos[0] <= self.hei) & (pos[1] >= 0) & (pos[1] <= self.wid)
        # pos = pos[mask,:]

        pos = np.swapaxes(np.swapaxes(pos,0,2),0,1)
        pos = pos[:, ::-1]
        # pos = pos[:,:,:]
        return pos
    
    def render(self, img: np.ndarray[int, np.dtype[np.int32]]):
        img = self.texture[:-1, :-1, :]
        return img
        # self.render_pts(img)
        # self.render_triangles()
    
    def render_pts(self, img: np.ndarray[int, np.dtype[np.int32]]):
        vec_x = self.pos[:,:,0].ravel().astype(np.int32)
        vec_y = self.pos[:,:,1].ravel().astype(np.int32)
        # filter out the points that are outside the texture
        mask = (vec_x >= 0) & (vec_x <= img.shape[0]) & (vec_y >= 0) & (vec_y <= img.shape[1])
        vec_x = vec_x[mask]
        vec_y = vec_y[mask]
        img[vec_x, vec_y, :] = 0

    def render_triangles(self):
        # on sweep de en haut à gauche à en bas à droite. sens de lecture.
        for row in range(self.pos.shape[0]):
            if row % 2 == 1:
                for col in range(self.pos.shape[1]-1):
                    # get points as tuples
                    pt_bot = tuple(self.pos[row, col, :])
                    pt_tl = tuple(self.pos[row-1, col, :])
                    pt_tr = tuple(self.pos[row-1, col+1, :])
                    # tk.render_debug_point(self.texture, pt_bot)
                    # tk.render_debug_point(self.texture, pt_tl)
                    # tk.render_debug_point(self.texture, pt_tr)
                    # while True:
                    #     tk.render(self.texture)
                    #     if cv2.waitKey(1) == ord('q'):
                    #         cv2.destroyAllWindows()
                    #         break
                    # check that the points are inside the image
                    if (pt_bot[0] >= 0 and pt_bot[0] < self.wid and
                        pt_tl[0] >= 0 and pt_tl[0] < self.wid and
                        pt_tr[0] >= 0 and pt_tr[0] < self.wid and
                        pt_bot[1] >= 0 and pt_bot[1]+1 < self.hei and
                        pt_tl[1] >= 0 and pt_tl[1] < self.hei and
                        pt_tr[1] >= 0 and pt_tr[1] < self.hei
                    ):
                        tk.triangle(
                            self.texture, 
                            pt_bot, 
                            pt_tl, 
                            pt_tr, 
                            (0, 0, 0), 
                            1
                        )
                        # flood fill the triangle
                        pt = tk.add_points(pt_bot, (0, 2))
                        pt = tk.tup_float2int(pt)
                        mask = tk.flood_fill_mask(self.texture, pt)
                        self.texture[mask] = 0
            elif row % 2 == 0 and row != 0:
                for col in range(1,self.pos.shape[1]-1):
                    # get points as tuples
                    pt_bot = tuple(self.pos[row, col, :])
                    pt_tl = tuple(self.pos[row-1, col-1, :])
                    pt_tr = tuple(self.pos[row-1, col, :])
                    # check that the points are inside the image
                    if (pt_bot[0] >= 0 and pt_bot[0] < self.wid and
                        pt_tl[0] >= 0 and pt_tl[0] < self.wid and
                        pt_tr[0] >= 0 and pt_tr[0] < self.wid and
                        pt_bot[1] >= 0 and pt_bot[1]+1 < self.hei and
                        pt_tl[1] >= 0 and pt_tl[1] < self.hei and
                        pt_tr[1] >= 0 and pt_tr[1] < self.hei
                    ):
                        tk.triangle(
                            self.texture, 
                            pt_bot, 
                            pt_tl, 
                            pt_tr, 
                            (0, 0, 0), 
                            1
                        )
                        pt = tk.add_points(pt_bot, (0, 2))
                        pt = tk.tup_float2int(pt)
                        mask = tk.flood_fill_mask(self.texture, pt)
                        self.texture[mask] = 0

    def add_to_pos(self, x:float):
        self.texture = np.roll(self.texture, -x, axis=1)
        # self.pos[:,:,0] -= x
        # self.pos[:,:,1] -= x
        # self.pos[:,:,1] %= self.hei



if __name__=="__main__":
    tex = HexTexture(400, 400)
    t = 0
    T = 1000
    img = np.ones((400, 400, 3), dtype = "uint8") * 255
    img = tex.render(img)
    tk.render(img)
    while t < T:
        # tk.render(img)
        # pass
        t+=1
        tex.add_to_pos(1)
        img = np.ones((400, 400, 3), dtype = "uint8") * 255
        img = tex.render(img)
        tk.render(img)
    #     # cv2.imshow("output", img)
        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            break
