from torch import nn
from ..quick import conv,convt
from ... import nn as gnn
class UNetBase(nn.Module):
    def __init__(self,ch=64):
        super().__init__()
        self.c1 = conv(1, ch, 3, 2, 1, act=nn.ReLU())
        self.c2 = conv(ch, ch*2, 3, 2, 1, act=nn.ReLU())
        self.c3 = conv(ch*2, ch*3, 3, 2, 1, act=nn.ReLU())
        self.c4 = conv(ch*3, ch*4, 3, 2, 1, act=nn.ReLU())

        self.tc1 = convt(ch*4, ch*3, 2, 2, act=nn.ReLU())
        self.tc2 = gnn.SignalConcat(convt(ch*6, ch*2, 2, 2, act=nn.ReLU()))
        self.tc3 = gnn.SignalConcat(convt(ch*4, ch, 2, 2, act=nn.ReLU()))
        self.tc4 = gnn.SignalConcat(convt(ch*2, 11, 2, 2))



    def forward(self, x1):
        x2 = self.c1(x1)
        x3 = self.c2(x2)
        x4 = self.c3(x3)
        x = self.c4(x4)

        x = self.tc1(x)
        x = self.tc2(x, x4)
        x = self.tc3(x, x3)
        x = self.tc4(x, x2)
        return x