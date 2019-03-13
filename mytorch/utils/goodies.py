import os
import time
import json
import torch
import pickle
import argparse
import warnings
import traceback
import numpy as np

from pathlib import Path
from collections import namedtuple
from torch.autograd import Function

TRACES_FORMAT = ['train_acc', 'train_loss', 'val_acc']

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


def default_eval(y_pred, y_true):
    """
        Expects a batch of input

        :param y_pred: tensor of shape (b, nc)
        :param y_true: tensor of shape (b, 1)
    """
    return torch.mean((torch.argmax(y_pred, dim=1) == y_true).float())


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

    def sorted(self, data=None):
        """
            If data given, then sort that and return, if not, sort self.
        :param dict: optional: dict
        :return:
        """
        if not data:
            data = self
        return [(x, data[x]) for x in sorted(data.keys(), key=lambda w: -data[w])]

    def cropped_with_freq(self, f):
        return sorted({tok: freq for tok, freq in self.items() if freq > f})


tosave = namedtuple('ObjectsToSave','fname obj')

def mt_save_dir(parentdir: Path, _newdir: bool = False):
    """
            Function which returns the last filled/or newest unfilled folder in a particular dict.
            Eg.1
                parentdir is empty dir
                    -> mkdir 0
                    -> cd 0
                    -> return parentdir/0

            Eg.2
                ls savedir -> 0, 1, 2, ... 9, 10, 11
                    -> mkdir 12 && cd 12 (if newdir is True) else 11
                    -> return parentdir/11 (or 12)

            ** Usage **
            Get a path using this function like so:
                parentdir = Path('runs')
                savedir = save_dir(parentdir, _newdir=True)

        :param parentdir: pathlib.Path object of the parent directory
        :param _newdir: bool flag to save in the last dir or make a new one
        :return: None
    """
    assert parentdir.is_dir(), f'{parentdir} is not a directory!'

    # Check if the dir exits
    assert parentdir.exists(), f'{parentdir} does not exist.'

    # List all folders within, and convert them to ints
    existing = sorted([int(x) for x in os.listdir(parentdir) if x.isdigit()], reverse=True)

    if not existing:
        # If no subfolder exists
        parentdir = parentdir / '0'
        parentdir.mkdir()
    elif _newdir:
        # If there are subfolders and we want to make a new dir
        parentdir = parentdir / str(existing[0] + 1)
        parentdir.mkdir()
    else:
        # There are other folders and we dont wanna make a new folder
        parentdir = parentdir / str(existing[0])

    return parentdir

def mt_save(savedir: Path, message: str= None, torch_stuff: list = None, pickle_stuff: list = None,
            numpy_stuff: list = None, json_stuff: list = None):
    """

        Saves bunch of diff stuff in a particular dict.

        NOTE: all the stuff to save should also have an accompanying filename, and so we use tosave named tuple defined above as
            tosave = namedtuple('ObjectsToSave','fname obj')

        ** Usage **
        # say `encoder` is torch module, and `traces` is a python obj (dont care what)
        parentdir = Path('runs')
        savedir = save_dir(parentdir, _newdir=True)
        save(
                savedir,
                torch_stuff = [tosave(fname='model.torch', obj=encoder)],
                pickle_stuff = [tosave('traces.pkl', traces)]
            )


    :param savedir: pathlib.Path object of the parent directory
    :param message: a message to be saved in the folder alongwith (as text)
    :param torch_stuff: list of tosave tuples to be saved with torch.save functions
    :param pickle_stuff: list of tosave tuples to be saved with pickle.dump
    :param numpy_stuff: list of tosave tuples to be saved with numpy.save
    :param json_stuff: list of tosave tuples to be saved with json.dump
    :return: None
    """

    assert savedir.is_dir(), f'{savedir} is not a directory!'

    # Commence saving shit!
    if message:
        with open('message.txt','w+') as f:
            f.write(message)

    for data in torch_stuff or ():
        try:
            torch.save(data.obj, savedir / data.fname)
        except:
            traceback.print_exc()

    for data in pickle_stuff or ():
        try:
            pickle.dump(data.obj, open(savedir / data.fname, 'wb+'))
        except:
            traceback.print_exc()

    for data in numpy_stuff or ():
        try:
            np.save(savedir / data.fname, data.obj)
        except:
            traceback.print_exc()

    for data in json_stuff or ():
        try:
            json.dump(data.obj, open(savedir / data.fname, 'w+'))
        except:
            traceback.print_exc()


def str2bool(v):
    """
        Function (copied from -https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse )

    :param v:
    :return:
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')