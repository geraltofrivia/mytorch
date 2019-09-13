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
from .utils.goodies import *
from . import dataiters


def simplest_loop(epochs: int,
                  data: dict,
                  opt: torch.optim,
                  loss_fn: torch.nn,
                  train_fn: Callable,
                  predict_fn: Callable,
                  device: torch.device = torch.device('cpu'),
                  data_fn: classmethod = dataiters.SimplestSampler,
                  eval_fn: Callable = default_eval) -> (list, list, list):
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
    valid_acc = []
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

                y_pred = train_fn(_x)
                loss = loss_fn(y_pred, _y)

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

                per_epoch_vl_acc.append(eval_fn(y_pred, _y).item())

        # Bookkeep
        train_acc.append(np.mean(per_epoch_tr_acc))
        train_loss.append(np.mean(per_epoch_loss))
        valid_acc.append(np.mean(per_epoch_vl_acc))

        print("Epoch: %(epo)03d | Loss: %(loss).5f | Tr_c: %(tracc)0.5f | Vl_c: %(vlacc)0.5f | Time: %(time).3f min"
              % {'epo': e,
                 'loss': float(np.mean(per_epoch_loss)),
                 'tracc': float(np.mean(per_epoch_tr_acc)),
                 'vlacc': float(np.mean(per_epoch_vl_acc)),
                 'time': timer.interval / 60.0})

    return train_acc, valid_acc, train_loss


def generic_loop(epochs: int,
                 data: int,
                 device: torch.device,
                 opt: torch.optim,
                 loss_fn: torch.nn,
                 model: torch.nn.Module,
                 train_fn: Callable,
                 predict_fn: Callable,
                 save: bool = False,
                 save_params: dict = None,
                 save_dir: Path = None,
                 save_above: float = -np.inf,
                 save_args: dict = None,
                 epoch_count: int = 0,
                 epoch_start_hook: Callable = None,
                 epoch_end_hook: Callable = None,
                 batch_start_hook: Callable = None,
                 batch_end_hook: Callable = None,
                 weight_decay: float = 0.0,
                 clip_grads_at: float = -1.0,
                 lr_schedule=None,
                 data_fn: classmethod = dataiters.SimplestSampler,
                 eval_fn: Callable = None,
                 notify: bool = False,
                 notify_key: str = None) -> (list, list, list, list):
    """

        A generic training loop, which based on diff hook fns (defined below), should handle anything given to it.

        The model need not be an nn.Module,
             but should have correctly wired forward and a predict function.

        # Data input
            Data should be a dict like so:
                {"train":{"x":np.arr, "y":np.arr}, "val":{"x":np.arr, "y":np.arr} }

        # Saving Logic
            If the flag is enabled, give in the dir and it'll save traces and the model (and the model encoder)
                everytime training acc exceeds all prev ones.

        ## If you want to save diff parts of the model,
        Prepare save args like -
            save_args = {'torch_stuff': [tosave('model.torch', clf.state_dict()), tosave('model_enc.torch', clf.encoder.state_dict())]}
        and pass it to the model alongwith.
        If the arg is empty, it defaults to -
            save_args = {'torch_stuff': [tosave('model.torch', model.state_dict())]}

    :param epochs: number of epochs to train for
    :param data: data dict (structure specified above)
    :param device: torch device to init the tensors with
    :param opt: torch optimizer, with proper param_groups for better lr decay per laye
    :param loss_fn: torch.nn loss fn
    :param model: torch module needed for
            i: grad clipping
            ii: for calling eval() and train() (regarding dropout)
    :param train_fn: a function which takes x & y, returns loss and y_pred
    :param predict_fn: a fn which takes x and returns y_pred
    :param save: [OPTIONAL] bool which wants either doesn't save, or saves at best
    :param save_dir: [OPTIONAL] Path object to which we save stuff (based on save_best)
    :param save_params: [OPTIONAL] a dict of all the params used while running and training the model.
    :param save_above: [OPTIONAL] acts as threshold regarading model saving. If the current trn accuracy is less than this, won't.
    :param save_args: [OPTIONAL] reference to the model to be saved
    :param epoch_count: an int which is added with #epochs (for better representation of how many epochs have actually passed).
            You can use this for when you run the loop say 3 times, do something else and run it for another 10.
    :param epoch_start_hook: a fn that can be called @ start of every epoch (returns model, opt)
    :param epoch_end_hook: a fn that can be called @ end of every epoch (returns model, opt)
    :param batch_start_hook: a fn that can be called @ start of every batch (returns model, opt)
    :param batch_end_hook: a fn that can be called @ end of every batch (returns model, opt)
    :param weight_decay: a L2 ratio (as mentioned in (https://arxiv.org/pdf/1711.05101.pdf)
    :param clip_grads_at: in case you want gradients clipped, send the max val here
    :param lr_schedule: a schedule that is called @ every batch start.
    :param data_fn: a class to which we can pass X and Y, and get an iterator.
    :param eval_fn: (optional) function which when given pred and true, returns acc
    :param notify: (optional) flag which enables sending notifications to your phones once the loop is done.
    :param notify_key: (optional) the api key to which the notification is to be sent. You can give it here, or in a file (see README.md)
    :return: traces
    """

    train_loss = []
    train_acc = []
    val_acc = []
    lrs = []
    saved_info = {}

    # Epoch level
    for e in range(epoch_count, epochs + epoch_count):

        per_epoch_loss = []
        per_epoch_tr_acc = []

        # Train
        with Timer() as timer:

            # Enable dropouts
            model.train()

            # @TODO: Add hook at start of epoch (how to decide what goes in)
            if epoch_start_hook: epoch_start_hook()

            # Make data
            trn_dl, val_dl = data_fn(data['train']), data_fn(data['valid'])

            for x, y in tqdm(trn_dl):

                if batch_start_hook: batch_start_hook()
                opt.zero_grad()

                if lr_schedule: lrs.append(update_lr(opt, lr_schedule.get()))

                _x = torch.tensor(x, dtype=torch.long, device=device)
                _y = torch.tensor(y, dtype=torch.long, device=device)

                y_pred = train_fn(_x)
                loss = loss_fn(y_pred, _y)

                per_epoch_tr_acc.append(eval_fn(y_pred=y_pred, y_true=_y).item())
                per_epoch_loss.append(loss.item())

                loss.backward()

                if clip_grads_at > 0.0: torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grads_at)
                for group in opt.param_groups:
                    for param in group['params']:
                        param.data = param.data.add(-weight_decay * group['lr'], param.data)

                opt.step()
                if batch_end_hook: batch_end_hook()

            if epoch_end_hook: epoch_end_hook()

        # Val
        with torch.no_grad():

            # Disable dropouts
            model.eval()

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

        # Save block (flag and condition)
        if save and train_acc[-1] >= save_above:
            # Update threshold
            save_above = train_acc[-1]

            # Adding epoch info along with options
            if save_params:
                save_params['epoch'] = e
            else:
                save_paras = {'epoch': e}

            # Prepare save_args if none
            if not save_args:
                save_args = {'torch_stuff': [tosave('model.torch', model.state_dict())]}

            # Call save function and save
            mt_save(save_dir,
                    torch_stuff=None if 'torch_stuff' in save_args.keys() else save_args['torch_stuff'],
                    pickle_stuff=[
                        tosave('traces.pkl', [train_acc, val_acc, train_loss, lrs]),
                        tosave('unsup_options.pkl', save_params)])
            print(f"Model saved on Epoch {e} at {save_dir} because of highest training acc so far")

            # Log the saved thing
            saved_info['epoch'] = e
            saved_info['accuracy'] = train_acc[-1]
            saved_info['directory'] = save_dir

    if notify:
        if not saved_info:
            message_template = "Your model is done training."
            send_notification(data=saved_info, key=notify_key, message_template=message_template)
        else:
            send_notification(data=saved_info, key=notify_key)

    return train_acc, val_acc, train_loss, lrs


# Let's write hooks to mimic phase 2 data prep
def reset_hidden(model, **args):
    for l in model.encoder.hidden:
        for h in l:
            h.data.zero_()

# epoch_start_hook_unsup_ft = partial(reset_hidden, model)
