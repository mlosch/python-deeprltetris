import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import random

_base36 = [ '0','1','2','3','4','5','6','7','8','9','A',
            'B','C','D','E','F','G','H','I','J','K','L',
            'M','N','O','P','Q','R','S','T','U','V','W',
            'X','Y','Z']

def num2base36(v):
    result = _base36[v % 36]

    while (v / 36) > 0:
        v /= 36
        result = _base36[v % 36] + result

    return result


def choosewithprob(p):
    dice = random.random() * sum(p)
    cum = 0

    for i in range(len(p)):
        if cum >= dice:
            return i

        cum += p[i]

    return len(p)-1


class Scoreplot(object):
    def __init__(self):
        self.scores = []
        self.x = []
        #plt.ion()
        self.fig = plt.figure()
        self.graph = plt.plot(self.x, self.scores)[0]
        plt.xlabel('Game')
        plt.ylabel('Avg Score')
        self.curve = animation.FuncAnimation(self.fig, self._updateline, 25, fargs=(self.graph,), interval=100, blit=True)
        #plt.show()

    def _updateline(self, num, line):
        line.set_ydata(self.scores)
        line.set_xdata(self.x)

    def newscore(self, game, score):
        self.scores.append(score)
        #self.x.append(len(self.scores))
        self.x.append(game)

    def updatescore(self, game, score):
        self.scores[game] = score

    def plot(self):
        pass
        # # plt.figure(self.fig.number)
        # #self.fig.clear()
        # #plt.xlabel('Game')
        # #plt.ylabel('Avg Score')
        # #plt.plot(self.x, self.scores)
        #
        # self.graph.set_ydata(self.scores)
        # self.graph.set_xdata(self.x)
        # plt.draw()
        # plt.pause(0.1)


# l, = plt.plot([], [], 'r-')
# plt.xlim(0, 1)
# plt.ylim(0, 1)
# plt.xlabel('x')
# plt.title('test')
# line_ani = animation.FuncAnimation(fig1, update_line, 25, fargs=(data, l),
#     interval=50, blit=True)
