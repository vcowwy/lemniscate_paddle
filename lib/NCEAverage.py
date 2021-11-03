from x2paddle import torch2paddle
import paddle
"""from torch.autograd import Function"""
from paddle.autograd import PyLayer
from paddle import nn
from .alias_multinomial import AliasMethod
import math


class NCEFunction(PyLayer):

    @staticmethod
    def forward(self, x, y, memory, idx, params):
        K = int(params[0].item())
        T = params[1].item()
        Z = params[2].item()
        momentum = params[3].item()
        batchSize = x.size(0)
        outputSize = memory.size(0)
        inputSize = memory.size(1)
        idx.select(1, 0).copy_(y.data)
        """weight = torch.index_select(memory, 0, idx.view(-1))"""
        weight = paddle.index_select(memory, 0, idx.view(-1))
        weight.resize_(batchSize, K + 1, inputSize)
        """out = torch.bmm(weight, x.data.resize_(batchSize, inputSize, 1))"""
        out = paddle.bmm(weight, x.data.resize_(batchSize, inputSize, 1))
        out.div_(T).exp_()
        x.data.resize_(batchSize, inputSize)
        if Z < 0:
            params[2] = out.mean() * outputSize
            Z = params[2].item()
            print('normalization constant Z is set to {:.1f}'.format(Z))
        out.div_(Z).resize_(batchSize, K + 1)
        self.save_for_backward(x, memory, y, weight, out, params)
        return out

    @staticmethod
    def backward(self, gradOutput):
        x, memory, y, weight, out, params = self.saved_tensors
        K = int(params[0].item())
        T = params[1].item()
        Z = params[2].item()
        momentum = params[3].item()
        batchSize = gradOutput.size(0)
        gradOutput.data.mul_(out.data)
        gradOutput.data.div_(T)
        gradOutput.data.resize_(batchSize, 1, K + 1)
        """gradInput = torch.bmm(gradOutput.data, weight)"""
        gradInput = paddle.bmm(gradOutput.data, weight)
        gradInput.resize_as_(x)
        weight_pos = weight.select(1, 0).resize_as_(x)
        weight_pos.mul_(momentum)
        weight_pos.add_(paddle.multiply(x.data, 1 - momentum))
        w_norm = weight_pos.pow(2).sum(1, keepdim=True).pow(0.5)
        updated_weight = weight_pos.div(w_norm)
        memory.index_copy_(0, y, updated_weight)
        return gradInput, None, None, None, None


class NCEAverage(nn.Layer):

    def __init__(self, inputSize, outputSize, K, T=0.07, momentum=0.5, Z=None):
        super(NCEAverage, self).__init__()
        self.nLem = outputSize
        self.unigrams = paddle.to_tensor(paddle.ones(self.nLem), stop_gradient=True)
        self.multinomial = AliasMethod(self.unigrams)
        self.K = K
        self.register_buffer('params', paddle.to_tensor([K, T, -1, momentum]))
        stdv = 1.0 / math.sqrt(inputSize / 3)
        self.register_buffer('memory', torch2paddle.rand(outputSize,
            inputSize).mul_(2 * stdv).add_(-stdv))

    def forward(self, x, y):
        batchSize = x.size(0)
        idx = self.multinomial.draw(batchSize * (self.K + 1)).view(batchSize,
            -1)
        out = NCEFunction.apply(x, y, self.memory, idx, self.params)
        return out
