import pyglet
from pyglet.window import key
import sys
from lib.Board import Board
from lib.Game import Game

BLOCK_IMG_FILE = 'img/block.png'

# these are the dimensions from the gameboy version
BOARD_WIDTH = 14
BOARD_HEIGHT = 20

block = pyglet.image.load(BLOCK_IMG_FILE)
block_sprite = pyglet.sprite.Sprite(block)

BLOCK_WIDTH = block.width
BLOCK_HEIGHT = block.height

window = pyglet.window.Window(width=BOARD_WIDTH*BLOCK_WIDTH,
                              height=BOARD_HEIGHT*BLOCK_HEIGHT)

board = Board(BOARD_WIDTH, BOARD_HEIGHT, block)

if len(sys.argv) > 1:
    starting_level = int(sys.argv[1])
else:
    starting_level = 1

game = Game(window, board, starting_level)

@window.event
def on_draw():
    game.draw_handler()

@window.event
def on_text_motion(motion):
    game.keyboard_handler(motion)

@window.event
def on_key_press(key_pressed, mod):
    if key_pressed == key.P:
        game.toggle_pause()

def update(dt):
    game.cycle()

pyglet.clock.schedule_interval(update, 1 / game.frame_rate)
pyglet.app.run()