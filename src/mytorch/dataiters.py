"""
    A file which contain simple data iterators.
"""
from .utils.goodies import *


class SimplestSampler:
    """
        Given X and Y matrices (or lists of lists),
            it returns a batch worth of stuff upon __next__
    :return:
    """

    def __init__(self, data, bs: int = 64):

        try:
            assert len(data["x"]) == len(data["y"])
        except AssertionError:

            raise MismatchedDataError(f"Length of x is {len(data['x'])} while of y is {len(data['y'])}")

        self.x = data["x"]
        self.y = data["y"]
        self.n = len(self.x)
        self.bs = bs  # Batch Size

    def __len__(self):
        return self.n // self.bs - (1 if self.n % self.bs else 0)

    def __iter__(self):
        self.i, self.iter = 0, 0
        return self

    def __next__(self):
        """
        @TODO: edge case: Return leftovers.
        :return:
        """
        if self.i + self.bs >= self.n:
            raise StopIteration

        _x, _y = self.x[self.i:self.i + self.bs], self.y[self.i:self.i + self.bs]
        self.i += self.bs

        return _x, _y


class SortishSampler:
    """
        Sample the data so like-sized text appears together.
        Returns an iterator.

        Works well with list
        @TODO: check/optimize for np arrs

        This needs to be re-sorted every iteration.
            Hence the data is duplicated internally.
            Call reset at epoch end to resort it.

            Or you could init a new instance of this :peace:
    """

    def __init__(self, data, _batchsize: int, _seqlen: int = None, _padidx=0):
        """ @TODO: snip everything with seqlen """
        _inputs, _labels = data['x'], data['y']

        try:
            assert len(_inputs) == len(_labels)
        except AssertionError:
            raise MismatchedDataError

        self.bs = _batchsize
        self.padidx = _padidx
        self.x, self.y = self._reshuffle_(**self._sort_(_inputs, _labels))

        self.ptr = 0

    def reset(self):
        self.x, self.y = self._reshuffle_(self.x, self.y)
        self.ptr = 0

    @staticmethod
    def _reshuffle_(x: list, y: list) -> (list, list):
        """
            Shuffles both, things inside a chunk (batch) and batches.
        :param x: list of np arr
        :param y: list of np arr
        :return: (list, list)
        """
        for i in range(len(x)):
            # Shuffle these chunks
            chunk_idx = np.random.permutation(len(x[i]))
            x[i] = x[i][chunk_idx]
            y[i] = y[i][chunk_idx]

        shuffle_idx = np.random.permutation(len(x))
        return [x[i] for i in shuffle_idx], [y[i] for i in shuffle_idx]

    def _sort_(self, x, y):
        idx = sorted(range(len(x)), key=lambda k: -len(x[k]))
        x, y = [x[i] for i in idx], [y[i] for i in idx]

        final_x, final_y = [], []
        for ptr in range(len(x))[::self.bs]:
            # Now take a snippet of x and y based the line below
            chunk_x = x[ptr: ptr + self.bs if ptr + self.bs < len(x) else len(x)]
            chunk_y = y[ptr: ptr + self.bs if ptr + self.bs < len(x) else len(x)]

            # Find snippet's max len
            chunk_len = len(chunk_x[0])

            # Pad and max np arr of this batch
            npx = pad_sequence(chunk_x, chunk_len, self.padidx)
            npy = np.asarray(chunk_y)

            # Shuffle x and y
            chunk_idx = np.random.permutation(len(chunk_x))
            npx, npy = npx[chunk_idx], npy[chunk_idx]

            # Append to final thing
            final_x.append(npx)
            final_y.append(npy)

        # Finally, shuffle final_x,y
        # final_idx = np.random.permutation(len(final_x))
        # return [final_x[i] for i in final_idx], [final_y[i] for i in final_idx]
        return {'x': final_x, 'y': final_y}

    def __iter__(self):
        return self

    def __next__(self):
        """ Iter over self.x, self.y """
        if self.ptr >= len(self.x):
            raise StopIteration
        _x, _y = self.x[self.ptr], self.y[self.ptr]
        self.ptr += 1
        return _x, _y

    def __len__(self):
        return len(self.x)


if __name__ == "__main__":
    X = np.random.randint(0, 100, (200, 4))
    Y = np.random.randint(0, 100, (200, 1))
    bsz = 19
    diter = SimplestSampler(X, Y, bsz)
