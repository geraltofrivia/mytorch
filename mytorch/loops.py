"""
    This file contains training loops, available as function calls.

    ## USAGE

    Typically, a training loop will want a train, a predict and a eval function;
        alongwith other args and data which dictate what the loop should train on, and for how long.
    See the documentation of `simplest loop` to see what it'd look like.

"""

from tqdm import tqdm
from typing import Callable

# Local imports
from mytorch import dataiters
from mytorch.utils.goodies import *


def simplest_loop(epochs: int,
                  data: dict,
                  device: torch.device,
                  opt: torch.optim,
                  loss_fn: torch.nn,
                  train_fn: Callable,
                  predict_fn: Callable,
                  data_fn: classmethod = dataiters.SimplestSampler,
                  eval_fn: Callable = None) -> (list, list, list):
    """
        A fn which can be used to train a language model.

        The model doesn't need to be an nn.Module,
            but have an eval (optional), a train and a predict function.

        Data should be a dict like so:
            {"train":{"x":np.arr, "y":np.arr}, "val":{"x":np.arr, "y":np.arr} }

        Train_fn must return both loss and y_pred

        :param epochs: number of epochs to train for
        :param data: a dict having keys train_x, test_x, train_y, test_y
        :param device: torch device to create new tensor from data
        :param opt: optimizer
        :param loss_fn: loss function
        :param train_fn: function to call with x and y
        :param predict_fn: function to call with x (test)
        :param data_fn: a class to which we can pass X and Y, and get an iterator.
        :param eval_fn: (optional) function which when given pred and true, returns acc
        :return: traces
    """

    train_loss = []
    train_acc = []
    val_acc = []
    lrs = []

    # Epoch level
    for e in range(epochs):

        per_epoch_loss = []
        per_epoch_tr_acc = []

        # Train
        with Timer() as timer:

            # Make data
            trn_dl, val_dl = data_fn(data['train']), data_fn(data['valid'])

            for x, y in tqdm(trn_dl):
                opt.zero_grad()

                _x = torch.tensor(x, dtype=torch.long, device=device)
                _y = torch.tensor(y, dtype=torch.long, device=device)

                loss, y_pred = train_fn(_x, _y, loss_fn)

                per_epoch_tr_acc.append(eval_fn(y_pred=y_pred, y_true=_y).item())
                per_epoch_loss.append(loss.item())

                loss.backward()
                opt.step()

        # Val
        with torch.no_grad():

            per_epoch_vl_acc = []
            for x, y in tqdm(val_dl):
                _x = torch.tensor(x, dtype=torch.long, device=device)
                _y = torch.tensor(y, dtype=torch.long, device=device)

                y_pred = predict_fn(_x)

                per_epoch_vl_acc.append(eval(y_pred, _y).item())

        # Bookkeep
        train_acc.append(np.mean(per_epoch_tr_acc))
        train_loss.append(np.mean(per_epoch_loss))
        val_acc.append(np.mean(per_epoch_vl_acc))

        print("Epoch: %(epo)03d | Loss: %(loss).5f | Tr_c: %(tracc)0.5f | Vl_c: %(vlacc)0.5f | Time: %(time).3f min"
              % {'epo': e,
                 'loss': float(np.mean(per_epoch_loss)),
                 'tracc': float(np.mean(per_epoch_tr_acc)),
                 'vlacc': float(np.mean(per_epoch_vl_acc)),
                 'time': timer.interval / 60.0})

    return train_acc, train_loss, val_acc


def generic_loop(epochs: int,
                 data: int,
                 device: torch.device,
                 opt: torch.optim,
                 loss_fn: torch.nn,
                 model: torch.nn.Module,
                 train_fn: Callable,
                 predict_fn: Callable,
                 epoch_start_hook: Callable = None,
                 epoch_end_hook: Callable = None,
                 batch_start_hook: Callable = None,
                 batch_end_hook: Callable = None,
                 weight_decay: float = 0.0,
                 clip_grads_at: float = -1.0,
                 lr_schedule=None,
                 data_fn: classmethod = dataiters.SimplestSampler,
                 eval_fn: Callable = None) -> (list, list, list):
    """

        A generic training loop, which based on diff hook fns (defined below), should handle anything given to it.

        The model doesn't need to be an nn.Module,
            but have an eval (optional), a train and a predict function.

        Data should be a dict like so:
            {"train":{"x":np.arr, "y":np.arr}, "val":{"x":np.arr, "y":np.arr} }

        Train_fn must return both loss and y_pred

    :param epochs: number of epochs to train for
    :param data: data dict (structure specified above)
    :param device: torch device to init the tensors with
    :param opt: torch optimizer, with proper param_groups for better lr decay per laye
    :param loss_fn: torch.nn loss fn
    :param model: torch module (for grad clipping)
    :param train_fn: a function which takes x & y, returns loss and y_pred
    :param predict_fn: a fn which takes x and returns y_pred
    :param epoch_start_hook: a fn that can be called @ start of every epoch (returns model, opt)
    :param epoch_end_hook: a fn that can be called @ end of every epoch (returns model, opt)
    :param batch_start_hook: a fn that can be called @ start of every batch (returns model, opt)
    :param batch_end_hook: a fn that can be called @ end of every batch (returns model, opt)
    :param weight_decay: a L2 ratio (as mentioned in (https://arxiv.org/pdf/1711.05101.pdf)
    :param clip_grads_at: in case you want gradients clipped, send the max val here
    :param lr_schedule: a schedule that is called @ every batch start.
    :param data_fn: a class to which we can pass X and Y, and get an iterator.
    :param eval_fn: (optional) function which when given pred and true, returns acc
    :return: traces
    """

    train_loss = []
    train_acc = []
    val_acc = []
    lrs = []

    # Epoch level
    for e in range(epochs):

        per_epoch_loss = []
        per_epoch_tr_acc = []

        # Train
        with Timer() as timer:

            # @TODO: Add hook at start of epoch (how to decide what goes in)
            if epoch_start_hook: epoch_start_hook()

            # Make data
            trn_dl, val_dl = data_fn(data['train']), data_fn(data['valid'])

            for x, y in tqdm(trn_dl):

                batch_start_hook()
                opt.zero_grad()

                if lr_schedule: lrs.append(update_lr(opt, lr_schedule.get()))

                _x = torch.tensor(x, dtype=torch.long, device=device)
                _y = torch.tensor(y, dtype=torch.long, device=device)

                loss, y_pred = train_fn(_x, _y, loss_fn)

                per_epoch_tr_acc.append(eval_fn(y_pred=y_pred, y_true=_y).item())
                per_epoch_loss.append(loss.item())

                loss.backward()

                if clip_grads_at > 0.0: torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grads_at)
                for group in opt.param_groups:
                    for param in group['params']:
                        param.data = param.data.add(-weight_decay * group['lr'], param.data)

                opt.step()
                batch_end_hook()

            if epoch_end_hook: epoch_end_hook()

        # Val
        with torch.no_grad():

            per_epoch_vl_acc = []
            for x, y in tqdm(val_dl):
                _x = torch.tensor(x, dtype=torch.long, device=device)
                _y = torch.tensor(y, dtype=torch.long, device=device)

                y_pred = predict_fn(_x)

                per_epoch_vl_acc.append(eval_fn(y_pred, _y).item())

        # Bookkeep
        train_acc.append(np.mean(per_epoch_tr_acc))
        train_loss.append(np.mean(per_epoch_loss))
        val_acc.append(np.mean(per_epoch_vl_acc))

        print("Epoch: %(epo)03d | Loss: %(loss).5f | Tr_c: %(tracc)0.5f | Vl_c: %(vlacc)0.5f | Time: %(time).3f min"
              % {'epo': e,
                 'loss': float(np.mean(per_epoch_loss)),
                 'tracc': float(np.mean(per_epoch_tr_acc)),
                 'vlacc': float(np.mean(per_epoch_vl_acc)),
                 'time': timer.interval / 60.0})

    return train_acc, train_loss, val_acc, lrs


# Let's write hooks to mimic phase 2 data prep
def reset_hidden(model):
    for l in model.encoder.hidden:
        for h in l:
            h.data.zero_()

# epoch_start_hook_unsup_ft = partial(reset_hidden, model)
