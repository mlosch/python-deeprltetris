import numpy as np


class Ringbuffer(list):
    def __init__(self, size):
        super(Ringbuffer, self).__init__()
        self.maxsize = size
        self.idx = 0

    def append(self, p_object):
        if self.__len__() < self.maxsize:
            super(Ringbuffer, self).append(p_object)
        else:
            self.__setitem__(self.idx, p_object)

        self.idx = (self.idx + 1) % self.maxsize

    def __delitem__(self, key):
        raise NotImplementedError()


class Replay(Ringbuffer):
    def sample(self, n):
        n = min(n, self.__len__())
        r = np.random.randint(0, self.__len__(), n)
        return [self[ri] for ri in r]

