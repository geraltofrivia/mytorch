import os
import time
import json
import torch
import pickle
import warnings
import numpy as np

from pathlib import Path
from collections import namedtuple
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


tosave = namedtuple('ObjectsToSave','fname obj')

def save(savedir: Path, torch_stuff: list = None, pickle_stuff: list = None,
         numpy_stuff: list = None, json_stuff: list = None, _newdir:bool=False):
    """
        Function you can call which will place your stuff in a subfolder within a dir properly.
        Eg.1
            savedir is empty dir
                -> mkdir 0
                -> cd 0
                -> save stuff (torch, pickle, and numpy stuff)

        Eg.2
            ls savedir -> 0, 1, 2, ... 9, 10, 11
                -> mkdir 12 && cd 12 (if newdir is True) else 11
                -> save stuff

        NOTE: all the stuff to save should also have an accompanying filename, and so we use tosave named tuple defined above as
            tosave = namedtuple('ObjectsToSave','fname obj')

        ** Usage **
        # say `encoder` is torch module, and `traces` is a python obj (dont care what)
        savedir = Path('runs')
        save(
                savedir,
                torch_stuff = [tosave(fname='model.torch', obj=encoder)],
                pickle_stuff = [tosave('traces.pkl', traces)],
                newdir=False
            )


    :param savedir: pathlib.Path object of the parent directory
    :param torch_stuff: list of tosave tuples to be saved with torch.save functions
    :param pickle_stuff: list of tosave tuples to be saved with pickle.dump
    :param numpy_stuff: list of tosave tuples to be saved with numpy.save
    :param json_stuff: list of tosave tuples to be saved with json.dump
    :param _newdir: bool flag to save in the last dir or make a new one
    :return: None
    """

    assert savedir.is_dir(), f'{savedir} is not a directory!'

    if not torch_stuff and not pickle_stuff and not numpy_stuff:
        warnings.warn(f"No objects given to save at {savedir}")

    # Check if the dir exits
    assert savedir.exists(), f'{savedir} does not exist.'

    # List all folders within, and convert them to ints
    existing = sorted([int(x) for x in os.listdir(savedir)], reverse=True)

    if not existing:
        # If no subfolder exists
        savedir = savedir / '0'
        savedir.mkdir()
    elif _newdir:
        # If there are subfolders and we want to make a new dir
        savedir = savedir / str(existing[0] + 1)
        savedir.mkdir()
    else:
        # There are other folders and we dont wanna make a new folder
        savedir = savedir / str(existing[0])

    # Commence saving shit!
    for data in torch_stuff:
        torch.save(data.obj, savedir / data.fname)

    for data in pickle_stuff:
        pickle.dump(data.obj, open(savedir / data.fname, 'wb+'))

    for data in numpy_stuff:
        np.save(savedir / data.fname, data.obj)

    for data in json_stuff:
        json.dump(data.obj, open(savedir / data.fname, 'w+'))
