import os
import time
import json
import torch
import pickle
import requests
import argparse
import warnings
import traceback
import numpy as np
from pathlib import Path
from collections import namedtuple
from torch.autograd import Function
from typing import List, Dict, Union, Any, Optional

TRACES_FORMAT = {name: i for i, name in enumerate(['train_acc', 'train_loss', 'val_acc'])}


class CustomError(Exception): pass
class MismatchedDataError(Exception): pass
class NotifyAPIKeyNotFoundError(Exception): pass
class NotifyMessageMismatchError(Exception): pass
class ImproperCMDArguments(Exception): pass
class UnknownSpacyLang(ValueError): pass


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


def pad_sequence(matrix_seq: Union[list, np.array], max_length: int = -1, padidx: Any = 0) -> np.array:
    """
        Pads 2D data.

        + Works with list of list as well as numpy matrix
        - Does not work with string inputs.

    :param matrix_seq: a matrix of list
    :param max_length: desired pad len (if not provided, will pad with max length in matrix_seq)
    :param padidx: the id with which to pad the data (could be anything)
    :return: a padded np array
    """

    if max_length < 0:
        max_length = max([len(x) for x in matrix_seq])

    pad_matrix = np.zeros((len(matrix_seq), max_length)) + padidx
    for i, arr in enumerate(matrix_seq):
        pad_matrix[i, :min(max_length, len(arr))] = arr[:min(max_length, len(arr))]

    return pad_matrix


def update_lr(opt: torch.optim, lrs: Union[int, float, list, np.array]) -> Union[int, float, list, np.array]:
    """ Updates lr of the opt. Give it one num for uniform update. Arr otherwise """

    if type(lrs) in [float, int]:
        for grp in opt.param_groups:
            grp['lr'] = lrs
    else:

        # Check for lens
        assert len(opt.param_groups) == len(lrs), f"Mismatch b/w param group ({len(opt.param_groups)}) " \
                                                  f"and lr list sizes ({len(lrs)})."
        for grp, lr in zip(opt.param_groups, lrs):
            grp['lr'] = lr

    return lrs


def make_opt(model, opt_fn: torch.optim, lr: float = 0.001):
    """
        Based on model.layers it creates diff param groups in opt.
    """
    assert hasattr(model, 'layers'), "The model does not have a layers attribute. Check TODO-URL for a how-to"
    return opt_fn([{'params': l.parameters(), 'lr': lr} for l in model.layers])


def default_eval(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
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
        :param data: dict
        :return:
        """
        if not data:
            data = self
        return [(x, data[x]) for x in sorted(data.keys(), key=lambda w: -data[w])]

    def cropped_with_freq(self, f):
        return sorted({tok: freq for tok, freq in self.items() if freq > f})


tosave = namedtuple('ObjectsToSave', 'fname obj')

def _is_serializable_(obj) -> bool:
    """ Check if the obj can be JSON serialized """
    try:
        json.dumps(obj)
        return True
    except TypeError:
        return False

def _filter_serializables_(data: dict) -> dict:
    seralizables = {}
    for key, val in data.items():
        if _is_serializable_(key):
            seralizables[key] = value
    return seralizables

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

    # Check if the dir exits
    try:
        assert parentdir.exists(), f'{parentdir} does not exist. Making it'
    except AssertionError:
        parentdir.mkdir(parents=True)

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

def mt_save(savedir: Path, message: str = None, message_fname: str = None, torch_stuff: list = None, pickle_stuff: list = None,
            numpy_stuff: list = None, json_stuff: List[tosave] = None):
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
                torch_stuff = [tosave(fname='model.torch', obj=encoder.state_dict()],
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
        with open(savedir / 'message.txt' if message_fname is None else savedir / message_fname, 'w+') as f:
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
        data_ = _filter_serializables_(data.obj)
        try:
            json.dump(data_obj, open(savedir / data.fname, 'w+'))
        except:
            traceback.print_exc()


def str2bool(v)->bool:
    """
        Function (copied from -https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse )

    :param v:
    :return:
    """
    if v.__class__ is bool:
        return v
    elif v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def compute_mask(t: Union[torch.Tensor, np.array], padding_idx=0):
    """
    compute mask on given tensor t
    :param t: either a tensor or a nparry
    :param padding_idx: the ID used to represented padded data
    :return: a mask of the same shape as t
    """
    if type(t) is np.ndarray:
        mask = np.not_equal(t, padding_idx)*1.0
    else:
        mask = torch.ne(t, padding_idx).float()
    return mask


def send_notification(data: dict, key: str = None, message_template: str = None, title: str = None) -> None:
    """
        Code which can send notification to a phone (using push.techulus.com).
        It tries to find api key from disk if not given as arg.

        And data is put on a template of message to send it to a phone

    :param data: a dictionary containing (i) epoch count (int); (ii) accuracy (float) and (iii) save directory (str)
    :param key: a str containing an API key (optional).
    :param message_template: a str with placeholders to put in data .
        Eg. 'Model saved at %(directory)s, after %(epoch)d epochs, having %(accuracy)f accuracy'
    :param title: a str which is the title of the notification
    :return: None
    """

    if not key:
        try:
            key = open('./push-techulus-key', 'r').read()
        except FileNotFoundError:
            raise NotifyAPIKeyNotFoundError("Couldn't find key to send a notification with")

    if not title:
        title = 'Bring out the champagne, your model is trained!'

    if not message_template:
        message_template = \
            'A model trained for %(epoch)d epochs, which achieved' \
            ' a %(accuracy)s percent accuracy, is now stored at %(directory)s'

    # Check if data fits the template
    try:
        message = message_template % data
    except (KeyError, TypeError) as e:
        raise NotifyMessageMismatchError(f"The data with keys {list(data.keys())} does not fit the message template.")

    url = "https://push.techulus.com/api/v1/notify"
    payload = json.dumps({"title": title, "body": message})

    headers = {
        'Content-Type': "application/json",
        'x-api-key': key,
    }

    try:
        _ = requests.request("POST", url, data=payload, headers=headers)
        print("Successfully delivered a notification on your cellphone. Cheers!")
    except:     # @TODO: Figure out which exceptions to catch
        traceback.print_exc()
        warnings.warn("Couldn't deliver notifications. Apologies. Report the traceback as an issue on Github, please?")


# Transparent, and simple argument parsing FTW!
def convert_nicely(arg, possible_types=(bool, float, int, str)):
    """ Try and see what sticks. Possible types can be changed. """
    for data_type in possible_types:
        try:

            if data_type is bool:
                # Hard code this shit
                if arg in ['T', 'True', 'true']: return True
                if arg in ['F', 'False', 'false']: return False
                raise ValueError
            else:
                proper_arg = data_type(arg)
                return proper_arg
        except ValueError:
            continue
    # Here, i.e. no data type really stuck
    warnings.warn(f"None of the possible datatypes matched for {arg}. Returning as-is")
    return arg


def parse_args(raw_args: List[str], compulsory: List[str] = (), compulsory_msg: str = "",
               types: Dict[str, type] = None, discard_unspecified: bool = False):
    """
        I don't like argparse.
        Don't like specifying a complex two liner for each every config flag/macro.

        If you maintain a dict of default arguments, and want to just overwrite it based on command args,
        call this function, specify some stuff like

    :param raw_args: unparsed sys.argv[1:]
    :param compulsory: if some flags must be there
    :param compulsory_msg: what if some compulsory flags weren't there
    :param types: a dict of confignm: type(configvl)
    :param discard_unspecified: flag so that if something doesn't appear in config it is not returned.
    :return:
    """

    parsed = {}

    while True:

        try:                                        # Get next value
            nm = raw_args.pop(0)
        except IndexError:                          # We emptied the list
            break

        # Get value
        try:
            vl = raw_args.pop(0)
        except IndexError:
            raise ImproperCMDArguments(f"A value was expected for {nm} parameter. Not found.")

        # Get type of value
        if types:
            try:
                parsed[nm] = types[nm](vl)
            except ValueError:
                raise ImproperCMDArguments(f"The value for {nm}: {vl} can not take the type {types[nm]}! ")
            except KeyError:                    # This name was not included in the types dict
                if not discard_unspecified:     # Add it nonetheless
                    parsed[nm] = convert_nicely(vl)
                else:                           # Discard it.
                    continue
        else:
            parsed[nm] = convert_nicely(vl)

    # Check if all the compulsory things are in here.
    for key in compulsory:
        try:
            assert key in parsed
        except AssertionError:
            raise ImproperCMDArguments(compulsory_msg + f"Found keys include {[k for k in parsed.keys()]}")

    # Finally check if something unwanted persists here
    return parsed
