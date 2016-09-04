import numpy as np
from pyglet.window import key
import random
import util
import math
import Replay

import theano
import theano.tensor as T
import lasagne
from lasagne.layers.dnn import Conv2DDNNLayer

import cPickle as pickle


class WorldFeedback(object):
    def getreward(self):
        raise NotImplementedError


class RLLearner(object):
    _moves = [key.MOTION_DOWN, key.MOTION_LEFT, key.MOTION_RIGHT, key.MOTION_UP]  # Do nothing, Move left, Move right

    def __init__(self, board, worldfeedback, learningrate, discountfactor, epsilon):
        self.board = board
        self.feedback = worldfeedback
        self.lr = learningrate
        self.gamma = discountfactor
        self.epsilon = epsilon

    def reset(self):
        pass

    def update(self, state, gameover, newpiece):
        pass

    def takeaction(self, state):
        return random.choice(self._moves)

    def dump(self, file):
        raise NotImplementedError()

    def load(self, file):
        raise NotImplementedError()


class QLearner(RLLearner):
    def __init__(self, board, worldfeedback, learningrate=0.01, discountfactor=0.6, epsilon=0.1):
        super(QLearner, self).__init__(board, worldfeedback, learningrate, discountfactor, epsilon)
        self.epsilon = epsilon

        self.lastState = None
        self.action = 0

        self.state = None

        self.reset()

        self.policy = {}

    def dump(self, f):
        pickle.dump(self.policy, open(f, 'w'), -1)

    def load(self, f):
        self.policy = pickle.load(open(f, 'r'))

    def _createpolicyentry(self, state):
        self.policy[state] = np.random.rand(len(self._moves))

    def reset(self):
        self.lastState = None
        self.state = None
        self.action = None

    def softmax(self, state):
        p = [math.exp(a) for a in self.policy[state]]
        s = sum(p)
        return [v / s for v in p]

    def _nextaction(self, state):
        # return util.choosewithprob(self.softmax(state))
        if random.random() < self.epsilon:
            return random.randint(0, len(self.policy[state]) - 1)
        else:
            return np.argmax(self.policy[state])

    def _updatevalue(self, gameover, state, reward):
        terminal = 0 if gameover else 1
        # Q-Learning always updates with the argmax action:
        nextAction = np.argmax(self.policy[state])
        self.policy[self.lastState][self.action] += self.lr * (
        reward + terminal * self.gamma * self.policy[state][nextAction] - self.policy[self.lastState][self.action])

        self.action = nextAction

    def takeaction(self, state):
        self.lastState = state

        if state not in self.policy:
            self._createpolicyentry(state)

        if self.action is None:
            # choose action according to state
            self.action = self._nextaction(state)

        return self._moves[self.action]

    def update(self, newstate, gameover, newpiece):

        reward = self.feedback.getreward()

        if newstate not in self.policy:
            self._createpolicyentry(newstate)

        # print(len(self.policy), state)
        # print(len(self.policy), self.policy[state], reward)

        self._updatevalue(gameover, newstate, reward)


class SarsaLearner(QLearner):
    def _updatevalue(self, gameover, state, reward):
        terminal = 0 if gameover else 1
        # SARSA-Learning always updates with the chosen action:
        nextAction = self._nextaction(state)
        self.policy[self.lastState][self.action] += self.lr * (
        reward + terminal * self.gamma * self.policy[state][nextAction] - self.policy[self.lastState][self.action])

        self.action = nextAction


class SarsaLambdaLearner(QLearner):
    """
    Implements SARSA with eligibility traces.
    """

    def __init__(self, board, worldfeedback, learningrate=0.01, discountfactor=0.6, epsilon=0.1, lam=0.9):
        self.lam = lam
        # self.e = {}
        self.track = []
        super(SarsaLambdaLearner, self).__init__(board, worldfeedback, learningrate, discountfactor, epsilon)

    def reset(self):
        super(SarsaLambdaLearner, self).reset()
        self.track = []

    def _updatevalue(self, gameover, state, reward):
        terminal = 0 if gameover else 1
        # SARSA-Learning always updates with the chosen action:
        nextAction = self._nextaction(state)
        delta = reward + terminal * self.gamma * self.policy[state][nextAction] - self.policy[self.lastState][
            self.action]

        hit = False
        for i in range(len(self.track)):
            if self.track[i][1] == self.action and self.track[i][0] == self.lastState:
                self.track[i][2] += 1
                hit = True
                break

        if not hit:
            self.track.append([self.lastState, self.action, 1])

        clearids = set()

        for i in range(len(self.track)):
            s, a, e = self.track[i]
            self.policy[s][a] += self.lr * delta * e
            self.track[i][2] *= self.gamma * self.lam

            if self.track[i][2] < 1e-6:
                clearids.add(i)

        if len(clearids) > 0:
            self.track = [t for i, t in enumerate(self.track) if i not in clearids]

        self.action = nextAction


class DeepQLearner(RLLearner):
    def __init__(self, board, worldfeedback, learningrate=0.01, discountfactor=0.6, epsilon=0.1, depsilon=0.0,
                 minepsilon=0.1, rho=0.99, rms_epsilon=1e-6, batchsize=32):
        super(DeepQLearner, self).__init__(board, worldfeedback, learningrate, discountfactor, epsilon)
        self.depsilon = depsilon
        self.minepsilon = minepsilon

        self.replaybuf = Replay.Replay(1000000)
        self.batchsize = batchsize

        last_state = T.tensor4('last_state')
        last_action = T.icol('last_action')
        state = T.tensor4('state')
        reward = T.col('reward')
        terminal = T.icol('terminal')

        self.state_shared = theano.shared(
            np.zeros((batchsize, 1, board.height, board.width), dtype=theano.config.floatX))
        self.last_state_shared = theano.shared(
            np.zeros((batchsize, 1, board.height, board.width), dtype=theano.config.floatX))
        self.last_action_shared = theano.shared(np.zeros((batchsize, 1), dtype='int32'), broadcastable=(False, True))
        self.reward_shared = theano.shared(np.zeros((batchsize, 1), dtype=theano.config.floatX),
                                           broadcastable=(False, True))
        self.terminal_shared = theano.shared(np.zeros((batchsize, 1), dtype='int32'), broadcastable=(False, True))

        model = lasagne.layers.InputLayer(shape=(batchsize, 1, board.height, board.width))
        model = Conv2DDNNLayer(model, 24, 3, pad=1, W=lasagne.init.HeUniform(), b=lasagne.init.Constant(.1))
        model = Conv2DDNNLayer(model, 48, 3, pad=1, W=lasagne.init.HeUniform(), b=lasagne.init.Constant(.1))
        model = Conv2DDNNLayer(model, 12, 3, pad=1, W=lasagne.init.HeUniform(), b=lasagne.init.Constant(.1))
        model = lasagne.layers.DenseLayer(model, 256, W=lasagne.init.HeUniform(), b=lasagne.init.Constant(.1))
        model = lasagne.layers.DenseLayer(model, len(self._moves), W=lasagne.init.HeUniform(),
                                          b=lasagne.init.Constant(.1),
                                          nonlinearity=lasagne.nonlinearities.identity)

        lastQvals = lasagne.layers.get_output(model, last_state)
        Qvals = lasagne.layers.get_output(model, state)
        Qvals = theano.gradient.disconnected_grad(Qvals)

        delta = reward + \
                terminal * self.gamma * T.max(Qvals, axis=1, keepdims=True) - \
                lastQvals[T.arange(batchsize), last_action.reshape((-1,))].reshape((-1, 1))

        loss = T.mean(0.5 * delta ** 2)

        params = lasagne.layers.get_all_params(model)
        givens = {
            state: self.state_shared,
            last_state: self.last_state_shared,
            last_action: self.last_action_shared,
            reward: self.reward_shared,
            terminal: self.terminal_shared,
        }
        updates = lasagne.updates.rmsprop(loss, params, learning_rate=self.lr, rho=rho, epsilon=rms_epsilon)

        self.model = model
        self.train_fn = theano.function([], [loss, Qvals], updates=updates, givens=givens)
        self.Qvals = theano.function([], Qvals, givens={state: self.state_shared})

        self.last_state = None
        self.action = None

        self.avgloss = []

    def dump(self, f):
        values = lasagne.layers.get_all_param_values(self.model)
        state = {'epsilon': self.epsilon, 'replay': self.replaybuf}
        pickle.dump({'values': values, 'state': state}, open(f, 'w'), -1)

    def load(self, f):
        data = pickle.load(open(f, 'r'))
        lasagne.layers.set_all_param_values(self.model, data['values'])
        self.replaybuf = data['state']['replay']
        self.epsilon = data['state']['epsilon']

    def takeaction(self, state):
        states = np.zeros((self.batchsize, 1, self.board.height,
                           self.board.width), dtype=theano.config.floatX)
        states[0, 0, ...] = state
        self.state_shared.set_value(states)

        if random.random() < self.epsilon:
            a = random.randint(0, len(self._moves) - 1)
        else:
            a = np.argmax(self.Qvals()[0])

        self.last_state = state
        self.action = a
        return self._moves[a]

    def getavgloss(self):
        avgloss = np.mean(self.avgloss)
        self.avgloss = []
        return avgloss

    def update(self, state, gameover, newpiece):
        terminal = 0 if gameover else 1
        reward = self.feedback.getreward()

        self.replaybuf.append((self.last_state, self.action, reward, state, terminal))

        # gather 100 samples before starting the training
        if len(self.replaybuf) >= 100:
            # do some actual training
            replay = self.replaybuf.sample(self.batchsize)
            W, H = self.board.width, self.board.height

            last_states = np.zeros((len(replay), 1, H, W), dtype=theano.config.floatX)
            states = np.zeros((len(replay), 1, H, W), dtype=theano.config.floatX)
            last_actions = np.zeros((len(replay), 1), dtype='int32')
            rewards = np.zeros((len(replay), 1), dtype=theano.config.floatX)
            terminals = np.zeros((len(replay), 1), dtype='int32')

            for i, r in enumerate(replay):
                last_states[i], last_actions[i], rewards[i], states[i], terminals[i] = replay[i]

            self.last_state_shared.set_value(last_states)
            self.state_shared.set_value(states)
            self.reward_shared.set_value(rewards)
            self.last_action_shared.set_value(last_actions)
            self.terminal_shared.set_value(terminals)

            loss, qvals = self.train_fn()

            self.avgloss.append(loss)

            if loss != loss:
                print(loss)

            self.epsilon = max(self.minepsilon, self.epsilon - self.depsilon)
