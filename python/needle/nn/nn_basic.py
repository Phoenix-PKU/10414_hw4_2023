"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(array = init.kaiming_uniform(
            in_features, out_features, nonlinearity="relu", device=device, dtype=dtype), 
            device=device, dtype=dtype)
        self.bias = None
        if bias:
            self.bias = Parameter(array = init.kaiming_uniform(
                out_features, 1, nonlinearity="relu", device=device, dtype=dtype).transpose(),
                device=device, dtype=dtype)       
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        output = ops.matmul(X, self.weight)
        if self.bias is not None:
            broadcasted_bias = ops.broadcast_to(self.bias, output.shape)
            output += broadcasted_bias
        return output
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        return ops.reshape(X, (X.shape[0], -1))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION

class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for module in self.modules:
            x = module(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        one_hot_vec = ops.summation(
            init.one_hot(logits.shape[-1], y, \
                device=logits.device, dtype=logits.dtype) * logits, axes=(1, ))
        return ops.summation(ops.logsumexp(logits, axes=(1, )) \
                - one_hot_vec, axes=(0, )) / logits.shape[0]
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.running_mean = init.zeros(dim, device=device, dtype=dtype)
        self.running_var = init.ones(dim, device=device, dtype=dtype)
        self.weight = Parameter(
            init.ones(dim, device=device, dtype=dtype), 
            device=device, dtype=dtype, requires_grad=True)  
        self.bias = Parameter(
            init.zeros(dim, device=device, dtype=dtype), 
            device=device, dtype=dtype, requires_grad=True)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        weight = self.weight.reshape((1, self.dim)).broadcast_to(x.shape)
        bias = self.bias.reshape((1, self.dim)).broadcast_to(x.shape)
        if self.training:
            e_x = ops.summation(x, axes = (0, )) / x.shape[0]
            self.running_mean = ((1 - self.momentum) * self.running_mean + \
                self.momentum * e_x).detach()
            e_x = e_x.reshape((1, x.shape[1])).broadcast_to(x.shape)

            var_x = ops.summation((x - e_x) ** 2, axes = (0, )) / x.shape[0]
            self.running_var = ((1 - self.momentum) * self.running_var + \
                self.momentum * var_x).detach()
            var_x = var_x.reshape((1, x.shape[1])).broadcast_to(x.shape)
            normed_x = (x - e_x) / (var_x + self.eps) ** 0.5
            return weight * normed_x + bias
        else:   
            normed_x = (x - self.running_mean) / (self.running_var + self.eps) ** 0.5
            return weight * normed_x + bias
        ### END YOUR SOLUTION



class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(
            dim, device=device, dtype=dtype, requires_grad=True), 
            device=device, dtype=dtype, requires_grad=True)
        self.bias = Parameter(init.zeros(
            dim, device=device, dtype=dtype, requires_grad=True), 
            device=device, dtype=dtype, requires_grad=True)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        e_x = ops.summation(x, axes = (1, )) / x.shape[1]
        e_x = e_x.reshape((x.shape[0], 1)).broadcast_to(x.shape)
        var_x = ops.summation((x - e_x) ** 2, axes = (1, )) / x.shape[1]
        var_x = var_x.reshape((x.shape[0], 1)).broadcast_to(x.shape)
        normed_x = (x - e_x) / (var_x + self.eps) ** 0.5
        weight = self.weight.reshape((1, self.dim)).broadcast_to(x.shape)
        bias = self.bias.reshape((1, self.dim)).broadcast_to(x.shape)
        return weight * normed_x + bias
        ### END YOUR SOLUTION


class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(
            dim, device=device, dtype=dtype, requires_grad=True), 
            device=device, dtype=dtype, requires_grad=True)
        self.bias = Parameter(init.zeros(
            dim, device=device, dtype=dtype, requires_grad=True), 
            device=device, dtype=dtype, requires_grad=True)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        e_x = ops.summation(x, axes = (1, )) / x.shape[1]
        e_x = e_x.reshape((x.shape[0], 1)).broadcast_to(x.shape)
        var_x = ops.summation((x - e_x) ** 2, axes = (1, )) / x.shape[1]
        var_x = var_x.reshape((x.shape[0], 1)).broadcast_to(x.shape)
        normed_x = (x - e_x) / (var_x + self.eps) ** 0.5
        weight = self.weight.reshape((1, self.dim)).broadcast_to(x.shape)
        bias = self.bias.reshape((1, self.dim)).broadcast_to(x.shape)
        return weight * normed_x + bias
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5, device=None, dtype="float32"):
        super().__init__()
        self.p = p
        self.device = device
        self.dtype = dtype

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # for p = 1, randb will return a tensor of all True
        # that means no dropout
        # however, we want p = 1 to mean all dropout
        zero_vector = init.randb(*x.shape, p = 1 - self.p, 
                    device=self.device, dtype=self.dtype)
        if self.training:
            return x * zero_vector / (1 - self.p)
        else:
            return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return x + self.fn(x)
        ### END YOUR SOLUTION
