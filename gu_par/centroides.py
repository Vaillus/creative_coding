

class Centroid:
    def __init__(
        self,
        x,
        y,
        w = 1
    ):
        self.x = x
        self.y = y
        self.w = w # mass
        # string composed of two directions: north-south, west-east
        # n : north
        # s : south
        # e : east
        # w : west
        # o : centre
        self.neighbours = {}
        self.valid_directions = ["ne","oe","se","sw","ow","nw"]
    
    def add_neighbour(self, dir, region):
        if dir in self.valid_directions:
            self.neighbours[dir] = region
        else:
            print("direction of the region not valid")
    
    def move(self):
        vecx = 0
        vecy = 0
        for nei in self.neighbours.values():
            vecx, vecy = self.update_mov_vec(nei, vecx, vecy)
        self.x += vecx
        self.y += vecy
    
    def update_mov_vec(self, nei, vecx, vecy):
        # to finish
        distx = nei.x - self.x
        disty = nei.y - self.y
        dist = sqrt(distx**2 + disty**2)
        theta = accos(distx / dist)
        F = self.w * nei.w / (dist ** 2)
        fx = F * cos(theta)
        fy = 
        vecx += fx
        vecy += fy
        return vecx, vecy


    
class Tiling:
    def __init__(
        self,
        n_row,
        n_col,
        w,
        h  
    ):
        self.n_row = n_row
        self.n_col = n_col
        self.grid = []
        self._init_grid(n_row, n_col, w , h)
    
    def _init_grid(self, n_row, n_col, w, h):
        self.grid = []
        for y in range(n_row):
            row = []
            for x in range(n_col):
                offset = self.get_offset(y)
                posx = w / n_row * (x + 1 + offset)
                posy = h / n_col * (y + 1)
                c = Centroid(posx, posy)
                row, c = self.connect_centroid(c, x, y, row)
                row = row + [c]
            self.grid += [row]
    
    def get_offset(self, y):
        if y % 2 == 0:
            return 0.5
        else:
            return 0.0
    
    def connect_centroid(self, c, col, row, rrow):
        if row % 2 == 0:
            # offset to the right every two rows.
            if col >= 1:
                c.add_neighbour("ow", rrow[col-1])
                rrow[col-1].add_neighbour("oe", c)
            if row >= 1:
                c.add_neighbour("nw", self.grid[row-1][col])
                self.grid[row-1][col].add_neighbour("se", c)
            if row >= 1 and col <= self.n_col-2:
                c.add_neighbour("ne", self.grid[row-1][col+1])
                self.grid[row-1][col+1].add_neighbour("sw", c)
        else:
            # offset to the left
            if col >= 1:
                c.add_neighbour("ow", rrow[col-1])
                rrow[col-1].add_neighbour("oe", c)
            if row >= 1 and col >=1:
                c.add_neighbour("nw", self.grid[row-1][col-1])
                self.grid[row-1][col-1].add_neighbour("se", c)
            if row >= 1:
                c.add_neighbour("ne", self.grid[row-1][col])
                self.grid[row-1][col].add_neighbour("sw", c)
        return rrow, c
    
    def update_up(self):
        for i in range(len(self.grid) - 1):
            self.grid[i] = self.grid[i+1]
        self.grid[-1] = self.init_new_row()
    
    def init_new_row(self):
        # I don't know where to place them.
        pass

    def centroids(self):
        for r in self.grid:
            for c in r:
                yield c

            
