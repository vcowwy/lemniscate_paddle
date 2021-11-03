import paddle
from paddle import nn
eps = 1e-07


class NCECriterion(nn.Layer):

    def __init__(self, nLem):
        super(NCECriterion, self).__init__()
        self.nLem = nLem

    def forward(self, x, targets):
        batchSize = x.size(0)
        K = x.size(1) - 1
        Pnt = 1 / float(self.nLem)
        Pns = 1 / float(self.nLem)
        Pmt = x.select(1, 0)
        Pmt_div = Pmt.add(K * Pnt + eps)
        """lnPmt = torch.div(Pmt, Pmt_div)"""
        lnPmt = paddle.divide(Pmt, Pmt_div)
        Pon_div = x.narrow(1, 1, K).add(K * Pns + eps)
        Pon = Pon_div.clone().fill_(K * Pns)
        """lnPon = torch.div(Pon, Pon_div)"""
        lnPon = paddle.divide(Pon, Pon_div)
        lnPmt.log_()
        lnPon.log_()
        lnPmtsum = lnPmt.sum(0)
        lnPonsum = lnPon.view(-1, 1).sum(0)
        loss = -(lnPmtsum + lnPonsum) / batchSize
        return loss
