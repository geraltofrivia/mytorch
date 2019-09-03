""" Testing different things we have put in in the utils file. """
# A bunch of imports
import torch
import torch.nn as nn

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
        yp = torch.randint(0, 3, (5, ))
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
            assert ((-1*grad) == grad_revved).all(), "The backward pass on a tensor does not 'just flip' the grads"
