""" Testing different things we have put in in the utils file. """
# A bunch of imports
import torch
import numpy as np
import torch.nn as nn
from typing import Any

# Testing code imports
from src.mytorch.utils import goodies as gd


class DummyNetwork(nn.Module):

    def __init__(self, n_cls: int):
        super().__init__()
        self.l = nn.Linear(5, n_cls)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.softmax(self.l(x), dim=1)

    def forward_gradrev(self, x: torch.Tensor) -> torch.Tensor:
        lin_op = self.l(x)
        return nn.functional.softmax(gd.GradReverse.apply(lin_op), dim=1)


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
