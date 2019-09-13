"""
    A collection of different learning rate iterators, that can be used to change the lr during training.

    The iterators ideally shouldn't be called on their own, but given to the LearningRateScheduler class
        along with optimizers, to better handle per-layer lr changing if needed.

    All lr iterators must have a coherent length.
"""
from .utils.goodies import *
from typing import Union, Type

class LearningRateSchedule:
    """ Empty class signifying that any derived class is a learning rate schedule. Purely cosmetic"""
    pass

class ConstantLR(LearningRateSchedule):
    def __init__(self, highest_lr: float, iterations: int):
        self.lr = highest_lr
        self.len = iterations

    def __iter__(self):
        return self

    def __next__(self):
        return self.lr

    def __len__(self):
        return self.len

    def reset(self):
        pass


class CosineAnnealingLR(LearningRateSchedule):
    """
        # Learning rate iter which decays w.r.t. cosine curve, #cycles times a batch.

        # USAGE:
            ```
                lr_schedule = CosineAnnealingLR(1000,3,0.01,0.001)
                a = [i for i in lr_schedule]
                plt.plot(a)
                plt.show()
            ```
    """

    def __init__(self, iterations: int, cycles: int, highest_lr: float, lowest_lr: float=0.0):
        """

        :param iterations: number of iters. Should be number of batches in one epoch.
        :param cycles: number of time the LR should restart in a batch
        :param highest_lr: The undecayed LR.
        :param lowest_lr: Limit to which the LR should decay.
        """

        self.iterations = iterations
        self.cycles = cycles
        self.lr_h = highest_lr
        self.lr_l = lowest_lr

        iters = self.iterations // self.cycles
        self.arr = (np.cos((np.arange(iters) / iters) * np.pi) / 2 + 0.5) * (self.lr_h - self.lr_l) + self.lr_l
        self.arr = np.tile(self.arr, cycles)

        self.ptr = -1

    def __iter__(self):
        return self

    def __len__(self)->int:
        return self.iterations

    def __next__(self)->float:
        """ Iter over self.arr """
        self.ptr += 1
        if self.ptr >= self.iterations:
            raise StopIteration

        if self.arr.shape[0] <= self.ptr < self.iterations:
            return self.lr_h
        else:
            return self.arr[self.ptr]

    def reset(self)->None:
        """ Reset the ptr """
        self.ptr = -1


class SlantedTriangularLR(LearningRateSchedule):
    """
        # An LR which first grows, and then decays back to zero w.r.t. iter (n_batch*n_epoch)

        # USAGE:
            ```
                lr_schedule = SlantedTriangularLR(1000,0.1,32,0.001)
                a = [i for i in lr_schedule]
                plt.plot(a)
                plt.show()
            ```
    """

    def __init__(self, iterations, cut_frac, ratio, highest_lr):
        """
        :param iterations: should be n_batch*n_epoch
        :param cut_frac: the fraction at which we should stop increasing and start decreasing LR
        :param ratio: another hyperparam deciding the shape of the entire thing.
        :param highest_lr: the undecayed LR.
        :param freeze_mask: is multiplied to the finally yielded values, can let some layers freeze, thus.
        """

        assert cut_frac < 1.0

        self.iter = iterations
        self.cf = cut_frac
        self.ratio = ratio
        self.lr_h = highest_lr
        self.cut = int(self.iter * self.cf)
        self.ptr = -1

    def __iter__(self):
        return self

    def __len__(self):
        return self.iter

    def __next__(self):
        """Iter over self.arr"""
        self.ptr += 1

        if self.ptr >= self.iter:
            raise StopIteration

        if self.ptr < self.cut:
            p = self.ptr / self.cut
        else:
            p = 1 - ((self.ptr - self.cut) / (self.cut * ((1 / self.cf) - 1)))

        return ((1 + float(p) * (self.ratio - 1)) / self.ratio * 1.0) * self.lr_h

    def reset(self):
        pass


class LearningRateScheduler:
    """
        An iterator which manages diff lr schedule iters, based on optimizer param groups.

        For each param_group in the optimzier,
            it will return a learning rate, sampled from given lr iterators.

        ```
            for g in opt.param_groups:
            g['lr'] = 0.1

            opt.param_groups[0]['lr'] = 0.0
            print([g['lr'] for g in opt.param_groups])

            lr = LearningRateScheduler(opt, epochs=10, batches=30)

            lr.get()   # Works for 300 (n_epochs * n_batches) times.
        ```
    """

    def __init__(self, lr_args: dict, lr_iterator: Type[LearningRateSchedule] = ConstantLR, org_lrs: list = None,
                 optimizer: torch.optim = None, freeze_mask: np.ndarray = None):
        """
        :param optimizer: torch.optim thing. Should have appropriate param groups for effective LR scheduling per layer.
        :param lr_args: a bunch of args (dict) intended for the given lr_iterator
        :param lr_iterator: a class reference of the intended lr schedule.
        """

        if optimizer is None and org_lrs is None:
            raise CustomError("No information specified to get highest lr.")
        elif optimizer is None:
            self.org_lrs = org_lrs
        else:
            self.opt = optimizer
            self.org_lrs = [group['lr'] for group in self.opt.param_groups]

        self.mask = np.array(freeze_mask) if freeze_mask is not None else np.ones_like(self.org_lrs)

        self.lr_iters = [lr_iterator(highest_lr=lr, **lr_args) for lr in self.org_lrs]

    def __len__(self):
        return min([len(_iter) for _iter in self.lr_iters])

    def get(self):
        try:
            return [lr_iter.__next__() for lr_iter in self.lr_iters] * self.mask
        except StopIteration:
            raise StopIteration(f"{self.__class__} was called more than the predefined times.")

    def reset(self)->None:
        """ To be called when the lr schedule needs to restart (e.g. at the end of every epoch). """
        for lr_iter in self.lr_iters:
            lr_iter.reset()

    def unfreeze(self):
        """ Find the last frozen layer and unfreeze it"""

        # First check if the mask isn't all ones
        if not (self.mask == 1).all():

            self.mask[np.max(np.where(self.mask == 0))] = 1
