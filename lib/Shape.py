import random


class Shape(object):
    """
    Change here the shapes that the agent is supposed to play with.
    """
    # # Default Tetris set:
    # _shapes = [
    #     [[0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]],   # I block
    #     [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]],   # O block
    #     [[0, 0, 0, 0], [0, 1, 0, 0], [1, 1, 1, 0], [0, 0, 0, 0]],   # T block
    #     [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 1, 1, 0]],   # J block
    #     [[0, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 1, 0]],   # L block
    #     [[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 1, 0], [0, 1, 0, 0]],   # S block
    #     [[0, 0, 0, 0], [0, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 0]],   # Z block
    # ]

    # # Only 2x2 block:
    # _shapes = [
    #     [[0,0,0,0], [0,1,1,0], [0,1,1,0], [0,0,0,0]],
    # ]

    # Only short L block:
    # _shapes = [
    #     [[0,0,0,0], [0,1,0,0], [0,1,1,0], [0,0,0,0]],
    # ]

    # # Collection of 2x2 blocks
    _shapes = [
        [[0,0,0,0], [0,1,0,0], [0,1,1,0], [0,0,0,0]],
        [[0,0,0,0], [0,1,1,0], [0,1,1,0], [0,0,0,0]],
        # [[0,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,0]],
        [[0,0,0,0], [0,0,1,0], [0,0,1,0], [0,0,0,0]],
    ]

    BLOCK_EMPTY = 0
    BLOCK_FULL = 1
    BLOCK_ACTIVE = 2

    def __init__(self, x=0, y=0):
        self.shapeidx = random.randint(0, len(self._shapes)-1)
        self.rotationidx = 0
        self.shape = self._shapes[self.shapeidx]
        self.shape = self.copy_shape()

        for i in range(random.choice(range(4))):
            self.rotate()

        self.x = x
        self.y = y

    def copy_shape(self):
        new_shape = []
        for row in self.shape:
            new_shape.append(row[:])
        return new_shape

    def clone(self):
        cloned = Shape()
        cloned.shape = self.copy_shape()
        cloned.rotationidx = self.rotationidx
        cloned.shapeidx = self.shapeidx
        cloned.x = self.x
        cloned.y = self.y
        return cloned

    def rotate(self):
        new_shape = self.copy_shape()
        for j in range(0, 4):
            for i in range(0, 4):
                new_shape[i][j] = self.shape[4 - j - 1][i]
        self.rotationidx = (self.rotationidx + 1) % 4
        self.shape = new_shape

    @property
    def left_edge(self):
        for x in range(0, 4):
            for y in range(0, 4):
                if self.shape[y][x] == Shape.BLOCK_FULL:
                    return x

    @property
    def right_edge(self):
        for x in range(3, -1, -1):
            for y in range(0, 4):
                if self.shape[y][x] == Shape.BLOCK_FULL:
                    return x

    @property
    def bottom_edge(self):
        for y in range(3, -1, -1):
            for x in range(0, 4):
                if self.shape[y][x] == Shape.BLOCK_FULL:
                    return y
