from Shape import Shape
import pyglet
from pyglet.window import key
from util import num2base36
import numpy as np


class Board(pyglet.event.EventDispatcher):
    active_shape = None
    pending_shape = None
    board = None
    encoding = None

    def __init__(self, width, height, block):
        self.width, self.height = width, height
        self.block = block
        self.calculated_height = self.height * block.height
        self.calculated_width = self.width * block.width
        self.reset()

    def reset(self):
        self.board = []
        for row in range(self.height):
            self.board.append([0] * self.width)

        self.pending_shape = Shape()
        self.add_shape()

    def add_shape(self):
        self.active_shape = self.pending_shape.clone()
        self.active_shape.x = self.width // 2 - self.active_shape.left_edge
        self.active_shape.y = -1
        self.pending_shape = Shape()

        if self.is_collision():
            self.reset()
            self.dispatch_event('on_game_over')

    def rotate_shape(self):
        rotated_shape = self.active_shape.clone()
        rotated_shape.rotate()

        if rotated_shape.left_edge + rotated_shape.x < 0:
            rotated_shape.x = -rotated_shape.left_edge
        elif rotated_shape.right_edge + rotated_shape.x >= self.width:
            rotated_shape.x = self.width - rotated_shape.right_edge - 1

        if rotated_shape.bottom_edge + rotated_shape.y > self.height:
            return False

        if not self.check_bottom(rotated_shape) and not self.is_collision(rotated_shape):
            self.active_shape = rotated_shape

    def move_left(self):
        self.active_shape.x -= 1
        if self.out_of_bounds() or self.is_collision():
            self.active_shape.x += 1
            return False
        return True

    def move_right(self):
        self.active_shape.x += 1
        if self.out_of_bounds() or self.is_collision():
            self.active_shape.x -= 1
            return False
        return True

    def move_down(self):
        self.active_shape.y += 1

        if self.check_bottom() or self.is_collision():
            self.active_shape.y -= 1
            self.shape_to_board()
            self.add_shape()
            return False
        return True

    def out_of_bounds(self, shape=None):
        shape = shape or self.active_shape
        if shape.x + shape.left_edge < 0:
            return True
        elif shape.x + shape.right_edge >= self.width:
            return True
        return False

    def check_bottom(self, shape=None):
        shape = shape or self.active_shape
        if shape.y + shape.bottom_edge >= self.height:
            return True
        return False

    def is_collision(self, shape=None):
        shape = shape or self.active_shape
        for y in range(4):
            for x in range(4):
                if y + shape.y < 0:
                    continue
                if shape.shape[y][x] and self.board[y + shape.y][x + shape.x]:
                    return True
        return False

    def test_for_line(self):
        for y in range(self.height - 1, -1, -1):
            counter = 0
            for x in range(self.width):
                if self.board[y][x] == Shape.BLOCK_FULL:
                    counter += 1
            if counter == self.width:
                self.process_line(y)
                return True
        return False

    def process_line(self, y_to_remove):
        for y in range(y_to_remove - 1, -1, -1):
            for x in range(self.width):
                self.board[y + 1][x] = self.board[y][x]

    def shape_to_board(self):
        # transpose onto board
        # while test for line, process & increase score
        for y in range(4):
            for x in range(4):
                dx = x + self.active_shape.x
                dy = y + self.active_shape.y
                if self.active_shape.shape[y][x] == Shape.BLOCK_FULL:
                    self.board[dy][dx] = Shape.BLOCK_FULL

        lines_found = 0
        while self.test_for_line():
            lines_found += 1

        if lines_found:
            self.dispatch_event('on_lines', lines_found)

        self.encoding = None

    def move_piece(self, motion_state):
        if motion_state == key.MOTION_LEFT:
            self.move_left()
        elif motion_state == key.MOTION_RIGHT:
            self.move_right()
        elif motion_state == key.MOTION_UP:
            self.rotate_shape()
        elif motion_state == key.MOTION_DOWN:
            self.move_down()

    def draw_game_board(self):
        for y, row in enumerate(self.board):
            for x, col in enumerate(row):
                if col == Shape.BLOCK_FULL or col == Shape.BLOCK_ACTIVE:
                    self.draw_block(x, y)

        for y in range(4):
            for x in range(4):
                dx = x + self.active_shape.x
                dy = y + self.active_shape.y
                if self.active_shape.shape[y][x] == Shape.BLOCK_FULL:
                    self.draw_block(dx, dy)

    def draw_block(self, x, y):
        y += 1 # since calculated_height does not account for 0-based index
        self.block.blit(x * self.block.width, self.calculated_height - y * self.block.height)

    def isoccupied(self, x, y):
        if self.board[y][x] == Shape.BLOCK_FULL or self.board[y][x] == Shape.BLOCK_ACTIVE:
            return True
        return False

    def encode_all(self):
        flags = []

        for y in range(4):
            for x in range(4):
                dx = x + self.active_shape.x
                dy = y + self.active_shape.y
                if self.active_shape.shape[y][x] == Shape.BLOCK_FULL:
                    self.board[dy][dx] = Shape.BLOCK_FULL

        for y in range(len(self.board)):
            for x in range(len(self.board[y])):
                if self.isoccupied(x, y):
                    flags.append('1')
                else:
                    flags.append('0')

        for y in range(4):
            for x in range(4):
                dx = x + self.active_shape.x
                dy = y + self.active_shape.y
                if self.active_shape.shape[y][x] == Shape.BLOCK_FULL:
                    self.board[dy][dx] = Shape.BLOCK_EMPTY

        return ''.join(flags)

    def encode_image(self):
        arr = np.asarray(self.board, dtype=np.float32)

        for y in range(4):
            for x in range(4):
                dx = x + self.active_shape.x
                dy = y + self.active_shape.y
                if self.active_shape.shape[y][x] == Shape.BLOCK_FULL:
                    arr[dy][dx] = Shape.BLOCK_FULL

        return arr

    def encode_only_static(self):
        flags = []
        for y in range(len(self.board)):
            for x in range(len(self.board[y])):
                if self.isoccupied(x, y):
                    flags.apped('1')
                else:
                    flags.append('0')
        return ''.join(flags)

    def encode_distance(self):
        d = [str(self.active_shape.shapeidx)]
        for x in range(len(self.board[0])):
            for y in range(len(self.board)):
                if self.isoccupied(x, y):
                    assert(y < 36)
                    d.append(num2base36(y))
                    break
            if len(d) <= x+1:
                d.append(num2base36(len(self.board)))
        return ''.join(d)

    def encode_toprows(self):

        if self.encoding is not None:
            self.encoding[0] = str(self.active_shape.shapeidx)
            self.encoding[1] = str(self.active_shape.rotationidx)
            self.encoding[2] = num2base36(self.active_shape.x + self.width // 2 - self.active_shape.left_edge)
        else:
            # encode active shape
            self.encoding = [str(self.active_shape.shapeidx), str(self.active_shape.rotationidx), num2base36(self.active_shape.x + self.width // 2 - self.active_shape.left_edge)]
            miny = len(self.board) - 3

            # find first empty row from the bottom up
            for y in range(len(self.board)-1, -1, -1):
                emptyrow = True
                for x in range(len(self.board[y])):
                    if self.isoccupied(x,y):
                        emptyrow = False
                        break

                if emptyrow:
                    miny = min(miny, y)
                    break

            # collect values
            for y in range(miny+1, miny+3):
                for x, cols in enumerate(self.board[y]):
                    if self.isoccupied(x, y):
                        self.encoding.append('1')
                    else:
                        self.encoding.append('0')

        return ''.join(self.encoding)

    def encode(self):
        return self.encode_toprows()

Board.register_event_type('on_lines')
Board.register_event_type('on_game_over')