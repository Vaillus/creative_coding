import numpy as np
import cv2

class HexTexture:
    def __init__(self, height: float, width: float):
        self.hei = height
        self.wid = width
        self.pos = self._make_reg_pos(10, 10)
        pass

    def _make_reg_pos(self, n_row:int, n_col:int):
        # initialize nonscaled positions relative to the upper left corner of 
        # the detection zone
        x_axis = np.array([[i for i in reversed(range(n_col))]] * n_row)
        y_axis = np.array([[i for i in reversed(range(n_row))]] * n_col).T
        # concatenate the x and z axes, then transform them in order to 
        # have a hexagonal distribution of the regions.
        pos = np.concatenate((np.expand_dims(x_axis, axis=0), np.expand_dims(
            y_axis, axis=0)), axis=0)
        pos = pos.astype(np.float64)
        pos[1] *= np.sqrt(3)/2
        pos[0,1::2] += 0.5
        pos[0] *= self.hei / n_col
        pos[1] *= self.wid / n_row
        pos = np.swapaxes(np.swapaxes(pos,0,2),0,1)
        pos = pos[:,:,:]
        return pos
    
    def print(self, img: np.ndarray[int, np.dtype[np.int32]]):
        vec_x = self.pos[:,:,0].ravel().astype(np.int32)
        vec_y = self.pos[:,:,1].ravel().astype(np.int32)
        img[vec_x, vec_y, :] = 0

    def add_to_pos(self, x:float):
        tex.pos[:,:,1] += x
        tex.pos[:,:,1] %= self.hei



if __name__=="__main__":
    tex = HexTexture(100, 100)
    
    t = 0
    T = 1000
    while t < T:
        t+=1
        tex.add_to_pos(1)
        img = np.ones((400, 400, 3), dtype = "uint8") * 255
        tex.print(img)
        cv2.imshow("output", img)
        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            break
