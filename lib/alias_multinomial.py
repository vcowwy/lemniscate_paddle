import paddle
import numpy as np


class AliasMethod(object):
    def __init__(self, probs):

        if probs.sum() > 1:
            probs.div_(probs.sum())
        K = len(probs)
        self.prob = paddle.to_tensor(paddle.zeros(K), stop_gradient=True)

        """self.alias = torch.LongTensor([0]*K)"""
        self.alias = paddle.to_tensor([0]*K, dtype=paddle.int64)

        smaller = []
        larger = []
        for kk, prob in enumerate(probs):
            self.prob[kk] = K * prob
            if self.prob[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)

        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()

            self.alias[small] = large
            self.prob[large] = self.prob[large] - 1.0 + self.prob[small]

            if self.prob[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

        for last_one in (smaller + larger):
            self.prob[last_one] = 1

    def draw(self, N):
        K = self.alias.size(0)

        kk = paddle.to_tensor(paddle.zeros(N, dtype=paddle.int64), stop_gradient=True).random_(0, K)

        """prob = self.prob.index_select(0, kk)"""
        """alias = self.alias.index_select(0, kk)"""
        """b = torch.bernoulli(prob)"""
        prob = paddle.index_select(self.prob, index=kk, axis=0)
        alias = paddle.index_select(self.alias, index=kk, axis=0)
        b = paddle.bernoulli(prob)

        oq = kk.mul(paddle.to_tensor(b, dtype=paddle.int64))
        oj = alias.mul(paddle.to_tensor((1 - b), dtype=paddle.int64))

        return oq + oj
