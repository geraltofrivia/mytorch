"""
    A file which contain simple data iterators.
"""
from utils.goodies import *


class SimplestDataIter:
    """
        Given X and Y matrices (or lists of lists),
            it returns a batch worth of stuff upon __next__
    :return:
    """

    def __init__(self, data, bs: int=64):

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
        if self.i + self.bs >= self.n:
            raise StopIteration

        _x, _y = self.x[self.i:self.i + self.bs], self.y[self.i:self.i + self.bs]
        self.i += self.bs

        return _x, _y


if __name__ == "__main__":
    X = np.random.randint(0, 100, (200, 4))
    Y = np.random.randint(0, 100, (200, 1))
    bsz = 19
    diter = SimplestDataIter(X, Y, bsz)
