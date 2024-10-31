from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND 

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        max_z_full = array_api.broadcast_to(
            Z.cached_data.max(axis = 1, keepdims = True), Z.shape)
        max_z = max(Z, axis = 1)
        log_sum_exp = array_api.log(array_api.sum(array_api.exp(Z - max_z_full), \
                        axis = 1)) + max_z
        log_sum_exp = array_api.broadcast_to(log_sum_exp.reshape((Z.shape[0], 1)), \
                        Z.shape)
        return Z - log_sum_exp
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]
        max_z_full = array_api.broadcast_to(
            Z.numpy().max(axis = 1, keepdims = True), Z.shape)
        sum_row_grad = summation(out_grad, axes = 1).reshape((Z.shape[0], 1))
        sum_exp = summation(exp(Z - max_z_full), axes = 1).reshape((Z.shape[0], 1))
        sum_matrix = (sum_row_grad / sum_exp).broadcast_to(Z.shape)
        return out_grad - sum_matrix * exp(Z - max_z_full)
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        if isinstance(axes, int):
            axes = (axes,)
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        max_z_full = array_api.broadcast_to(
            Z.max(axis = self.axes, keepdims = True), Z.shape)
        max_z = Z.max(axis = self.axes)
        return array_api.log(array_api.sum(array_api.exp(Z - max_z_full), \
                        axis = self.axes)) + max_z
        ### END YOUR SOLUTIONx

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        target_shape = node.inputs[0].shape
        new_shape = list(target_shape)
        axes = range(len(new_shape)) if self.axes is None else self.axes
        for axis in sorted(axes):
            new_shape[axis] = 1
        new_shape = tuple(new_shape)

        Z = node.inputs[0]
        max_z_full = array_api.broadcast_to(
            Z.cached_data.max(axis = 1, keepdims = True), Z.shape)

        sum_exp = summation(exp(Z - max_z_full), axes = self.axes).\
            reshape(new_shape).broadcast_to(target_shape)
  
        return out_grad.reshape(new_shape).broadcast_to(target_shape) \
                    * exp(Z - max_z_full) / sum_exp
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

