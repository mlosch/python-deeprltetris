from Learning import WorldFeedback


class Game(object):
    ticks = 0
    factor = 4
    frame_rate = 60.0

    is_paused = False

    def __init__(self, window_ref, board, starting_level=1):
        self.window_ref = window_ref
        self.board = board
        self.starting_level = int(starting_level)
        self.register_callbacks()
        self.reset()

    def register_callbacks(self):
        self.board.push_handlers(self)

    def reset(self):
        self.level = self.starting_level
        self.lines = 0
        self.score = 0

    def should_update(self):
        if self.is_paused:
            return False

        self.ticks += 1
        if self.ticks >= (self.frame_rate - (self.level * self.factor)):
            self.ticks = 0
            return True
        return False

    def draw_handler(self):
        self.window_ref.clear()
        self.board.draw_game_board()

    def keyboard_handler(self, motion):
        self.board.move_piece(motion)

    def on_lines(self, num_lines):
        self.score += (num_lines * self.level)
        self.lines += num_lines
        if self.lines / 10 > self.level:
            self.level = self.lines / 10

    def on_game_over(self):
        self.reset()

    def cycle(self):
        if self.should_update():
            self.board.move_down()
            if self.window_ref:
                self.update_caption()

    def toggle_pause(self):
        self.is_paused = not self.is_paused

    def update_caption(self):
        self.window_ref.set_caption('Tetris - %s lines [%s]' % (self.lines, self.score))


class RLGame(Game, WorldFeedback):

    """
    Change REWARD_LINE to give a reward when a line is complete.
    Change REWARD_GAMEOVER to give a negative reward when the topmost line has reached the game ceiling
    """
    REWARD_LINE = 1.0
    REWARD_GAMEOVER = 0  # -10.0


    _rewardcarriage = 0
    _gamecounter = -1  # -1 as .reset() is called once on construction which increments this counter

    def getreward(self):
        reward = self._rewardcarriage
        self._rewardcarriage = 0
        return reward - 1

    def should_update(self):
        self.ticks += 1
        return True

    def on_lines(self, num_lines):
        Game.on_lines(self, num_lines)
        self._rewardcarriage += num_lines*self.REWARD_LINE

    def manualreset(self):
        Game.reset(self)
        self._gamecounter += 1
        self._rewardcarriage = 0

    def reset(self):
        Game.reset(self)
        self._gamecounter += 1

    def on_game_over(self):
        Game.on_game_over(self)
        self._rewardcarriage += self.REWARD_GAMEOVER
