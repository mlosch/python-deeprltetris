import pyglet
from pyglet.window import key
from lib.Board import Board
from lib.Game import RLGame
from lib.Learning import QLearner, SarsaLambdaLearner, DeepQLearner
from lib.util import Scoreplot
import cPickle as pickle
import glob


"""
Things to play around with:
Checkout the following files for further adjustable parameters: Shape.py, Game.py, Learning.py

Change DRAW to True, to see the agent play.
Change the UPDATE_INTERVAL (in seconds) to define how many moves the agent can make per second.
Change PLOT to True, to draw a plot showing the average reward over games.

Change BOARD_WIDTH and BOARD_HEIGHT to define the size of the game board.

Change learner (in the middle of this file) to define which algorithm is used ( currently either Q or SARSA(lambda) )
"""
DRAW = False
UPDATE_INTERVAL = 0.1
PLOT = False

BOARD_WIDTH = 6
BOARD_HEIGHT = 12


global lastgame
global lastscores
lastgame = 0
lastscores = [0]

iteration = 0
maxscore = 0

BLOCK_IMG_FILE = 'img/block.png'

ACTIONNAMES = {
    -1: 'Do nothing',
    key.MOTION_LEFT: 'Move Left',
    key.MOTION_RIGHT: 'Move Right',
    key.MOTION_UP: 'Rotate',
    key.MOTION_DOWN: 'Move Down',
}

block = pyglet.image.load(BLOCK_IMG_FILE)
block_sprite = pyglet.sprite.Sprite(block)

BLOCK_WIDTH = block.width
BLOCK_HEIGHT = block.height

if DRAW:
    window = pyglet.window.Window(width=BOARD_WIDTH*BLOCK_WIDTH,
                              height=BOARD_HEIGHT*BLOCK_HEIGHT)
else:
    window = None

board = Board(BOARD_WIDTH, BOARD_HEIGHT, block)
if PLOT:
    plot = Scoreplot()

game = RLGame(window, board, 1)
# learner = SarsaLearner(board, game, learningrate=0.1, epsilon=0.0, discountfactor=0.7)
# learner = QLearner(board, game, learningrate=0.1, epsilon=0.05, discountfactor=0.7)
# learner = SarsaLambdaLearner(board, game, learningrate=0.1, epsilon=0.0, lam=0.9, discountfactor=0.7)
# boardencoding = board.encode

learner = DeepQLearner(board, game, learningrate=0.0002, discountfactor=0.95, epsilon=1.0, minepsilon=0, depsilon=1e-8)
boardencoding = board.encode_image

global state
state = boardencoding()

# Load an existing policy if available
files = glob.glob('policy-*.pickle')
if files and len(files) > 0:
    print('Loading policy from file: '+files[-1])
    file = files[-1]
    file = file[file.find('-')+1:file.find('.')]
    game._gamecounter = int(file)
    learner.load(files[-1])

if DRAW:
    @window.event
    def on_draw():
        if DRAW:
            game.draw_handler()

    @window.event
    def on_text_motion(motion):
        game.keyboard_handler(motion)

    @window.event
    def on_key_press(key_pressed, mod):
        if key_pressed == key.P:
            game.toggle_pause()


def update(dt):

    global state

    # choose action
    action = learner.takeaction(state)

    # perform the action on the board
    board.move_piece(action)

    # update game
    game.cycle()

    # encode board into next state
    state = boardencoding()


    gameover = False
    newpiece = False

    # collect statistics
    global lastgame
    global lastscores

    if game.lines > 0 and game.lines != lastscores[-1]:
        # announce new piece to learner
        newpiece = True

    if game.lines > 0:
        if game.lines != lastscores[-1]:
            lastscores[-1] = game.lines

            maxscorebound = 1000
            if lastscores[-1] >= maxscorebound:
                # reset game
                print('Scored %d in game %d. Starting next game.'%(lastscores[-1], lastgame))
                game.manualreset()
                board.reset()
                learner.reset()

    if game._gamecounter > lastgame:

        gameover = True

        lastgame = game._gamecounter
        if game._gamecounter % 100 == 0:
            avgscore = sum(lastscores)/float(len(lastscores))
            if PLOT:
                plot.newscore(lastgame, avgscore)
                plot.plot()

            print('Game: %d, Min score: %d, Max score: %d, Avg score: %f, Eps: %.3f, Avg. loss: %.5f'%(game._gamecounter, min(lastscores), max(lastscores), avgscore, learner.epsilon, learner.getavgloss()))
            lastscores = []
        lastscores.append(0)
    # plot.plot()

    # update learner
    learner.update(state, newpiece, gameover)

# import cProfile, pstats, cStringIO
# pr = cProfile.Profile()

try:
    if DRAW:
        pyglet.clock.schedule_interval(update, UPDATE_INTERVAL)
        pyglet.app.run()
    else:
        # pr.enable()
        while 1:
            update(0)
except KeyboardInterrupt:
    # pr.disable()
    # s = cStringIO.StringIO()
    # sortby = 'cumulative'
    # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    # ps.print_stats()
    # with open('profile_stats.txt', 'w') as f:
    #     f.write(s.getvalue())

    print('Interrupted')
    # save policy
    filename = 'policy-%d.pickle'%game._gamecounter
    learner.dump(filename)
    print('Policy saved to: '+filename)
