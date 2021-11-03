from x2paddle import torch2paddle
import paddle
"""from torch.autograd import Function"""
from paddle.autograd import PyLayer
from paddle import nn
import math


class LinearAverageOp(PyLayer):

    @staticmethod
    def forward(self, x, y, memory, params):
        T = params[0].item()
        batchSize = x.size(0)
        """out = torch.mm(x.data, memory.t())"""
        out = paddle.mm(x.data, memory.t())
        out.div_(T)
        self.save_for_backward(x, memory, y, params)
        return out

    @staticmethod
    def backward(self, gradOutput):
        x, memory, y, params = self.saved_tensors
        batchSize = gradOutput.size(0)
        T = params[0].item()
        momentum = params[1].item()
        gradOutput.data.div_(T)
        """gradInput = torch.mm(gradOutput.data, memory)"""
        gradInput = paddle.mm(gradOutput.data, memory)
        gradInput.resize_as_(x)
        """weight_pos = memory.index_select(0, y.data.view(-1)).resize_as_(x)"""
        weight_pos = memory.index_select(0, y.data.view(-1)).resize_as_(x)
        weight_pos.mul_(momentum)
        weight_pos.add_(paddle.multiply(x.data, 1 - momentum))
        w_norm = weight_pos.pow(2).sum(1, keepdim=True).pow(0.5)
        updated_weight = weight_pos.div(w_norm)
        memory.index_copy_(0, y, updated_weight)
        return gradInput, None, None, None


class LinearAverage(nn.Layer):

    def __init__(self, inputSize, outputSize, T=0.07, momentum=0.5):
        super(LinearAverage, self).__init__()
        stdv = 1 / math.sqrt(inputSize)
        self.nLem = outputSize
        self.register_buffer('params', paddle.to_tensor([T, momentum]))
        stdv = 1.0 / math.sqrt(inputSize / 3)
        self.register_buffer('memory', torch2paddle.rand(outputSize,
            inputSize).mul_(2 * stdv).add_(-stdv))

    def forward(self, x, y):
        out = LinearAverageOp.apply(x, y, self.memory, self.params)
        return out
