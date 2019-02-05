import time
import numpy as np
import torch
from torch.autograd import Function


class CustomError(Exception): pass
class MismatchedDataError(Exception): pass
class BadParameters(Exception):
    def __init___(self, dErrorArguments):
        Exception.__init__(self, "Unexpected value of parameter {0}".format(dErrorArguments))
        self.dErrorArguments = dErrorArguments


class FancyDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)
        self.__dict__ = self


class GradReverse(Function):
    """
        Torch function used to invert the sign of gradients (to be used for argmax instead of argmin)
        Usage:
            x = GradReverse.apply(x) where x is a tensor with grads.
    """
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()


def pad_sequence(matrix_seq, max_length, padidx=0):
    """
        Works with list of list as well as numpy matrix

    :param matrix_seq: a matrix of list
    :param max_length: desired pad len
    :param padidx: the id with which to pad the data
    :return:
    """

    pad_matrix = np.zeros((len(matrix_seq), max_length)) + padidx
    for i, arr in enumerate(matrix_seq):
        pad_matrix[i, :min(max_length, len(arr))] = arr[:min(max_length, len(arr))]

    return pad_matrix


def update_lr(opt: torch.optim, lrs) -> None:
    """ Updates lr of the opt. Give it one num for uniform update. Arr otherwise """

    if type(lrs) is float:
        for grp in opt.param_groups:
            grp['lr'] = lrs
    else:
        for grp, lr in zip(opt.param_groups, lrs):
            grp['lr'] = lr

    return lrs


def make_opt(model, opt_fn, lr=0.001):
    """
        Based on model.layers it creates diff param groups in opt.
    """
    return opt_fn([{'params': l.parameters(), 'lr': lr} for l in model.layers])


class Timer:
    """ Simple block which can be called as a context, to know the time of a block. """
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.interval = self.end - self.start


class Counter(dict):
    """ Assumes a list of data (words?), and counts their occurrence. """

    def __init__(self, data):
        super().__init__()

        for datum in data:
            self[datum] = self.get(datum, 0) + 1

    def most_common(self, n):
        return [(x, self[x]) for x in sorted(self.keys(), key=lambda w: -self[w])[:n]]

    def sorted(self):
        return [(x, self[x]) for x in sorted(self.keys(), key=lambda w: -self[w])]
