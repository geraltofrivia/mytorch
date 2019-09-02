""" Testing different things we have put in in the utils file. """
import torch
import torch.nn as nn

# Testing code imports
from src.mytorch.utils import goodies as gd


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
        _ = gd.GradReverse()

    def test_forward(self):
        """ See if the model does not alter the forward """
        gradrev = gd.GradReverse()

        ip = torch.randn(100, 100)
        op = gradrev(ip)

        ...

    def test_backward(self):
        """ See if the model does alter the grads in backward """
        ...

    def test_parameters(self):
        """ See if the thing has any parameters """
        ...

    def test_fits_in_module(self):
        """ Make a model, put grad rev in there and see if something funny happens. """
        ...

    def test_realistic(self):
        """ Make model, pass input, collect gradients. See if it reverses them. """