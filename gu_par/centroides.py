

class Centroid:
    def __init__(
        self,
        x,
        y
    ):
        self.x = x
        self.y = y
    
class Tiling:
    def __init__(
        self,
        n_row,
        n_col    
    ):
        self.n_row = n_row
        self.n_col = n_col
        self.grid = self._init_grid(n_row, n_col)
    
    def _init_grid(self, n_row, n_col):
        grid = []
        for i in range(n_row):
            row = []
            for j in range(n_col):
                row = row + [Centroid(0,0)]
            grid += [row]
        return grid
    
    def update_up(self):
        for i in range(len(self.grid) - 1):
            self.grid[i] = self.grid[i+1]
        self.grid[-1] = self.init_new_row()
    
    def init_new_row(self):
        # I don't know where to place them.
        pass

            
