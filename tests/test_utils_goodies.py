""" Testing different things we have put in in the utils file. """
# A bunch of imports
import torch
import numpy as np
import torch.nn as nn
from typing import Any

# Testing code imports
from mytorch.utils import goodies as gd


class DummyNetwork(nn.Module):

    def __init__(self, n_cls: int):
        super().__init__()
        self.la = nn.Linear(5, 10)
        self.lb = nn.Linear(10, n_cls)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.softmax(self.lb(self.la(x)), dim=1)

    def forward_gradrev(self, x: torch.Tensor) -> torch.Tensor:
        lin_op = self.lb(self.la(x))
        return nn.functional.softmax(gd.GradReverse.apply(lin_op), dim=1)

    @property
    def layers(self):
        return torch.nn.ModuleList([
            self.la, self.lb
        ])


class LongerNetwork(nn.Module):
    def __init__(self, cls: int = 3, lin_nm:int = 5):
        super().__init__()
        self.lins = [nn.Linear(5, 5) for _ in range(lin_nm)]
        self.clf = nn.Linear(5, cls)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for lin in self.lins:
            x = lin(x)

        return self.clf(x)

    @property
    def layers(self):
        return torch.nn.ModuleList(self.lins + [self.clf])


class TestFancyDict:
    """ Tests for the fancy dict. """

    def test_init(self):
        """ See if it inits """
        pass

    def test_append(self):
        """ Make a empty dict, throw things to it and see if it sticks. """
        fd = gd.FancyDict()
        fd['newkey'] = 'newval'

        assert 'newkey' in fd, "FancyDict is immutable! :o"
        assert fd['newkey'] is 'newval', "FancyDict does not maintain values."

    def test_access(self):
        """ Test different ways in which we can call a dict """
        fd = gd.FancyDict(newkey="newval", nextkey="nextval")

        assert fd['newkey'], "FancyDict does not like being accessed like all dicts."
        assert fd.nextkey, "FancyDict is no longer fancy."


class TestGradReverse:
    """ Tests to see if Grad Reverse works as it should """

    def test_init(self):
        """ See if class inits """
        try:
            _ = gd.GradReverse()
        except Exception as e:
            raise AssertionError(f"Error: {e} raised when trying to init grad reverse.")

    @staticmethod
    def get_dummy_network(n_cls: int = 5) -> DummyNetwork:
        """ Make a random net and throw in the layer in there """

        dummynet = DummyNetwork(n_cls)
        return dummynet

    def test_apply(self):
        """ See if the GradRev function can be applied """
        ip = torch.randn(5, 5)
        op = gd.GradReverse.apply(ip)

        assert (op == ip).all(), "The Grad Reverse function alters the input. It should not."

    def test_forward(self):
        """ See if the model does not alter the forward """
        net = self.get_dummy_network()
        ip = torch.randn(5, 5)

        try:
            _ = net.forward_gradrev(ip)
        except Exception as e:
            raise AssertionError(f"The forward pass on a random tensor returns the error: {e}")

    def test_backward(self):
        """ See if the model does alter the grads in backward """
        net = self.get_dummy_network(n_cls=3)
        ip = torch.randn(5, 5)
        yp = torch.randint(0, 3, (5,))
        loss_fn = nn.modules.loss.CrossEntropyLoss()

        # Get loss and backward
        op = net(ip)
        loss = loss_fn(op, yp)

        try:
            loss.backward()
        except Exception as e:
            raise AssertionError(f"The backward pass on a tensor returns the error {e}")

        # Grads
        grads = [param.grad.numpy().copy() for param in net.parameters()]

        # Run the input again
        net.zero_grad()
        op = net(ip)
        op = gd.GradReverse.apply(op)
        loss = loss_fn(op, yp)
        loss.backward()

        grads_revved = [param.grad.numpy() for param in net.parameters()]

        # Compare the two grads
        for grad, grad_revved in zip(grads, grads_revved):
            assert ((-1 * grad) == grad_revved).all(), "The backward pass on a tensor does not 'just flip' the grads"


class TestPadSequence:
    """ Test the pad sequences function """

    @staticmethod
    def get_manually_padded_2d(dtype: Any = int):
        if dtype is int or dtype is np.int or dtype is torch.float:
            return np.array([
                [1, 5, 4, 3, 0],
                [1, 2, 4, 0, 9],
                [9, 3, 0, 0, 0],
                [9, 4, 3, 0, 0]
            ])
        elif dtype is float or dtype is np.float or dtype is torch.float:
            return np.array([
                [1.2, 5.6, 4.1, 3.4, 0],
                [1, 2, 4.4, 0.2, 9.0],
                [9, 3, 0, 0, 0],
                [9, 4, 3, 0, 0]
            ])
        elif dtype is bool or dtype is np.bool or dtype is torch.bool:
            return np.array([
                [True, True, True, False, False],
                [False, False, True, False, False],
                [True, False, False, True, True],
                [True, False, False, False, False]
            ])

    def test_list(self):
        """ Check if list of lists can be padded """
        data = [
            [1, 5, 4, 3],
            [1, 2, 4, 0, 9],
            [9, 3],
            [9, 4, 3]
        ]

        padded_data = gd.pad_sequence(data)

        assert (
                padded_data == self.get_manually_padded_2d()).all(), "Pad sequence did not pad list of lists correctly."

    def test_nparray(self):
        """ Check if a np array can be padded """
        data = np.random.randint(0, 10, (4, 5))
        padded = gd.pad_sequence(data)

        assert (padded == data).all(), "Pad sequence did not pad np matrix as intended."

    def test_nparray_oflists(self):
        """ Check if a np array composed of python lists can be padded"""
        data = np.array([
            [1, 5, 4, 3],
            [1, 2, 4, 0, 9],
            [9, 3],
            [9, 4, 3]
        ])

        assert data.dtype is np.dtype('O'), "The test was not wired up properly"

        padded_data = gd.pad_sequence(data)
        assert (
                padded_data == self.get_manually_padded_2d()).all(), "Pad sequence did not pad np arr of lists as intended."

    def test_list_ofnparrays(self):
        data = [
            np.array([1, 5, 4, 3]),
            np.array([1, 2, 4, 0, 9]),
            np.array([9, 3]),
            np.array([9, 4, 3])
        ]

        padded_data = gd.pad_sequence(data)
        assert (
                padded_data == self.get_manually_padded_2d()).all(), "Pad sequences did not pad list of nparrs as intended."

    def test_float(self):
        """ See if a list of floats work """
        data = [
            [1.2, 5.6, 4.1, 3.4],
            [1, 2, 4.4, 0.2, 9.0],
            [9, 3],
            [9, 4, 3]
        ]
        assert (self.get_manually_padded_2d(float) == gd.pad_sequence(data)).all(), "A list of list of floats dont work"
        assert (self.get_manually_padded_2d(float) == gd.pad_sequence(np.array(data))).all(), \
            "A nparr of list of floats dont work"

    def test_bool(self):
        """ See if a list of bools work"""
        data = [
            [True, True, True],
            [False, False, True, False],
            [True, False, False, True, True],
            [True]
        ]
        assert (self.get_manually_padded_2d(bool) == gd.pad_sequence(data)).all(), "A list of list of bools dont work"

        data = np.array(data)
        assert (self.get_manually_padded_2d(bool) == gd.pad_sequence(data)).all(), "A nparr of list of bools dont work"

    def test_lesser_len(self):
        """ See if a list of list can be padded (with reduced len) """
        data = [
            [1, 5, 4, 3],
            [1, 2, 4, 0, 9],
            [9, 3],
            [9, 4, 3]
        ]
        padded_data = gd.pad_sequence(data, max_length=4)
        manually_padded_data = np.array([
            [1, 5, 4, 3],
            [1, 2, 4, 0],
            [9, 3, 0, 0],
            [9, 4, 3, 0]
        ])
        assert (manually_padded_data == padded_data).all(), \
            "Cannot properly pad when max length is less than the longest arr in data."

    def test_greater_len(self):
        """ See if both a list of list can be padded (with greater len) """
        data = [
            [1, 5, 4, 3],
            [1, 2, 4, 0, 9],
            [9, 3],
            [9, 4, 3]
        ]
        padded_data = gd.pad_sequence(data, max_length=6)
        manually_padded_data = np.array([
            [1, 5, 4, 3, 0, 0],
            [1, 2, 4, 0, 9, 0],
            [9, 3, 0, 0, 0, 0],
            [9, 4, 3, 0, 0, 0]
        ])
        assert (manually_padded_data == padded_data).all(), \
            "Cannot properly pad when max length is greater than the longest arr in data."

    def test_custom_padid(self):
        """ When we try a different value than default of padid, the thing works """
        data = [
            [1, 5, 4, 3],
            [1, 2, 4, 0, 9],
            [9, 3],
            [9, 4, 3]
        ]
        padded_data = gd.pad_sequence(data, padidx=-10)
        manually_padded_data = [
            [1, 5, 4, 3, -10],
            [1, 2, 4, 0, 9],
            [9, 3, -10, -10, -10],
            [9, 4, 3, -10, -10]
        ]
        assert (manually_padded_data == padded_data).all(), \
            " Does not work when pad id is not zero "

    def test_custom_padid_type(self):
        """ See if fn can handle float padidx when arr is int """
        int_data = [
            [1, 5, 4, 3],
            [1, 2, 4, 0, 9],
            [9, 3],
            [9, 4, 3]
        ]
        padded_int_data = gd.pad_sequence(int_data, padidx=3.9)
        manual_padded_int_data = np.array([
            [1, 5, 4, 3, 3.9],
            [1, 2, 4, 0, 9],
            [9, 3, 3.9, 3.9, 3.9],
            [9, 4, 3, 3.9, 3.9]
        ])
        assert (manual_padded_int_data == padded_int_data).all(), \
            "Arr not converted to float when padid is float but data is int"

        float_data = [
            [1.2, 5.6, 4.1, 3.4],
            [1, 2, 4.4, 0.2, 9.0],
            [9, 3],
            [9, 4, 3]
        ]
        padded_float_data = gd.pad_sequence(float_data, padidx=3)
        manual_padded_float_data = np.array([
            [1.2, 5.6, 4.1, 3.4, 3.0],
            [1, 2, 4.4, 0.2, 9.0],
            [9, 3, 3.0, 3.0, 3.0],
            [9, 4, 3, 3.0, 3.0]
        ])
        assert (manual_padded_float_data == padded_float_data).all(), \
            "Problem when padidx is int and data is float."

    def test_custom_padid_fails(self):
        """ See if a weird padid is passed, the code crashes. """
        data = [
            [1, 5, 4, 3],
            [1, 2, 4, 0, 9],
            [9, 3],
            [9, 4, 3]
        ]
        try:
            padded_data = gd.pad_sequence(data, padidx='potato')
            raise AssertionError("The function is expected to throw a type error when padidx is weird. It did not.")
        except TypeError:
            ...


class TestMakeOpt:
    """ Tests for updating learning rates """

    def test_init(self):
        """ Simplest use case """

        # Make a network
        net = DummyNetwork(3)
        optim_fn = torch.optim.SGD
        try:
            _ = gd.make_opt(net, optim_fn)
        except Exception as e:
            raise AssertionError(f"Creating an optimizer with this function fails with exception: {e}")

    def test_without_layers(self):
        """ Check if an assertion error is raised when model don't have layers """

        class NoLayerNet(nn.Module):

            def __init__(self):
                super().__init__()

                self.lina = nn.Linear(5, 10)
                self.linb = nn.Linear(10, 2)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.linb(self.lina(x))

        nolayernet = NoLayerNet()

        try:
            _ = gd.make_opt(nolayernet, torch.optim.SGD)
            raise AssertionError("When a model with no layers is passed, an assertion error was expected.")
        except AssertionError:
            ...
        except Exception as e:
            print("Some other error raised!")
            raise AssertionError(f"When a model with no layers is passed, an assertion error was expected."
                                 f"Got the following Exception instead: {e}")

    def test_param_groups(self):
        """ Check if param groups are made. Expected 2 for dummy net. """
        # Make a network
        net = DummyNetwork(3)
        optim_fn = torch.optim.SGD

        optim = gd.make_opt(net, optim_fn)

        assert optim.param_groups.__len__() > 1, "Optimizer is expected to have two param groups here."


class TestUpdateLR:
    """ Test if the updating learning rate based on param grouped optimizer works. """

    def test_init(self):
        """ Test if the function can be called in the simplest setting """
        net = DummyNetwork(3)
        optim = gd.make_opt(net, torch.optim.SGD)
        lr = 0.1

        try:
            _ = gd.update_lr(optim, lr)
        except Exception as e:
            raise AssertionError(f"The fn fails in the simplest setting with the exception being: {e}")

    def test_float_lr(self):
        """ Test if a float lr can update the optim """
        net = DummyNetwork(3)
        optim = gd.make_opt(net, torch.optim.SGD)
        lr = 0.1234323421

        gd.update_lr(optim, lr)

        for param_grp in optim.param_groups:
            assert param_grp['lr'] == lr, f"Cant reliably update with float lrs. Expected {lr}, got {param_grp['lr']}"

    def test_int_lr(self):
        """ Test if a int lr can update the optim """
        """ Test if a float lr can update the optim """
        net = DummyNetwork(3)
        optim = gd.make_opt(net, torch.optim.SGD)
        lr = 23

        gd.update_lr(optim, lr)

        for param_grp in optim.param_groups:
            assert param_grp['lr'] == lr, f"Cant reliably update with float lrs. Expected {lr}, got {param_grp['lr']}"

    def test_list_lr(self):
        """ Test if we can pass a list corresponding to the layers """
        net = DummyNetwork(3)
        optim = gd.make_opt(net, torch.optim.SGD)
        lr = [0.2, 0.4]

        gd.update_lr(optim, lr)

        for i, param_grp in enumerate(optim.param_groups):
            assert param_grp['lr'] == lr[i], f"Cant reliably update with float lrs. " \
                                             f"Expected {lr[i]}, got {param_grp['lr']}"

    def test_nparr_lr(self):
        """ Test if can pass an ndarray of the same len as that of layers """
        net = DummyNetwork(3)
        optim = gd.make_opt(net, torch.optim.SGD)
        lr = np.random.randn(2)

        gd.update_lr(optim, lr)

        for i, param_grp in enumerate(optim.param_groups):
            assert param_grp['lr'] == lr[i], f"Cant reliably update with float lrs. " \
                                             f"Expected {lr[i]}, got {param_grp['lr']}"

    def test_list_smaller(self):
        """ What happens when less lrs than param groups """
        net = LongerNetwork(lin_nm=5)
        optim = gd.make_opt(net, torch.optim.SGD)
        lr = [0.1, 0.3, 0.1, 2]

        try:
            gd.update_lr(optim, lr)
            raise AssertionError("Expected an assertion error when a smaller length is passed to the fn")
        except AssertionError:
            ...

    def test_list_larger(self):
        """ When happens when more lrs than param groups """
        net = LongerNetwork(lin_nm=5)
        optim = gd.make_opt(net, torch.optim.SGD)
        lr = [0.1, 0.3, 0.1, 2]

        try:
            gd.update_lr(optim, lr)
            raise AssertionError("Expected an assertion error when a smaller length is passed to the fn")
        except AssertionError:
            ...


class TestDefaultEval:
    """ Can it truly return accuracy? """

    def test_init(self):
        """ Does this work? """
        y_true = torch.randint(0, 5, (10,))
        y_pred = torch.randn(10, 5)

        try:
            _ = gd.default_eval(y_pred=y_pred, y_true=y_true)
        except Exception as e:
            raise AssertionError(f"This fn doesn't work. Exception received: {e}")

    def test_perfmatch(self):
        """ When acc is supposed to be one """
        yp = torch.tensor([
          [0.1, 0.2, 0.3, 0.1, 0.9],
          [0.4, 0.1, -0.3, 0.05, 0.3],
          [0.01, 0.3, 0.19, 0.9, 0.5],
          [0.2, 0.5, 0.8, 0.94, 0.2]
        ])
        yt = torch.tensor([4, 0, 3, 3])

        acc = gd.default_eval(y_pred=yp, y_true=yt)

        assert acc == 1.0, f"Received acc {acc}, Expected 1.0"

    def test_halfmatch(self):
        """ When acc is supposed to be half """
        yp = torch.tensor([
          [0.1, 2.2, 0.3, 0.1, 0.9],
          [0.4, 0.1, -0.3, 0.05, 0.3],
          [0.01, 0.3, 0.19, 0.9, 0.5],
          [0.2, 1.5, 0.8, 0.94, 0.2]
        ])
        yt = torch.tensor([4, 0, 3, 3])

        acc = gd.default_eval(y_pred=yp, y_true=yt)

        assert acc == 0.5, f"Received acc {acc}, Expected 0.5"

    def test_nomatch(self):
        """ When acc is supposed to be zero """
        yp = torch.tensor([
            [0.1, 2.2, 0.3, 0.1, 0.9],
            [0.4, 0.1, 1.3, 0.05, 0.3],
            [0.01, 9.3, 0.19, 0.9, 0.5],
            [0.2, 1.5, 0.8, 0.94, 0.2]
        ])
        yt = torch.tensor([4, 0, 3, 3])

        acc = gd.default_eval(y_pred=yp, y_true=yt)

        assert acc == 0.0, f"Received acc {acc}, Expected 0.0"


class TestTimer:
    """ Tests for the context thing which finds run time of a codeblock """

    def test_init(self):
        ...

    def test_sleep(self):
        """ use some os sleep thing """
        ...

    def test_empty(self):
        """ Empty context block """
        ...

    def test_mid_access(self):
        """ Can you get interval within the context """
        ...

    def test_nested(self):
        """ Test by nesting these things a level or two """
        ...

    def test_sequential(self):
        """ Make a timer. Make one again, and see if the interval still makes sense """
        ...


class TestCounter:
    """ """