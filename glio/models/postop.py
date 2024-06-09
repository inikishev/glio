import torch
from torch import nn
from monai.networks.nets import HighResNet, SegResNetDS # type:ignore
from ..python_tools import Compose, identity

def HighResNet_noReLUattheend(in_channels=12, out_channels=20):
    model = HighResNet(2, in_channels, out_channels)
    model.blocks[-1].adn = nn.Identity()
    return model

batch_tfms = identity

mse = torch.nn.MSELoss()

class ContextHRN(nn.Module):
    def __init__(self, pretraining = False):
        super().__init__()
        self.pretraining = pretraining
        self.batch_tfms = batch_tfms
        self.hrn = HighResNet_noReLUattheend(12, 20)

    def forward(self, x:torch.Tensor):
        if self.training and self.pretraining:
            with torch.no_grad(): self.tfmed = self.batch_tfms(x)
            self.processed = self.hrn(self.tfmed)
            loss = mse(self.processed[:, :12], x)
            return self.processed[:,12:16], loss
        elif self.pretraining: 
            self.processed = self.hrn(x)
            return self.processed[:,12:16]
        else:
            self.processed = self.hrn(x)
            return self.processed

class ContextSegResNetDS_FineTune(nn.Module):
    def __init__(self):
        super().__init__()
        self.context_block = ContextHRN(False)
        self.net = SegResNetDS(2, 32, 32, 4)
    def forward(self, x:torch.Tensor):
        self.context = self.context_block(x)
        return self.net(torch.cat((x, self.context), 1))


def get_model():
    model = ContextSegResNetDS_FineTune().to(torch.device("cuda"))
    model.load_state_dict(torch.load(r"F:\Stuff\Programming\experiments\vkr\training\RHUH v2 full + BRATS v2 test, 32\models\ft int+flip, contextnet-SegResNetDS+adan+dicefocal lr1e-03 OneCycleLR\model.pt"))
    return model

def get_learner():
    from ..train2 import Learner
    model = ContextSegResNetDS_FineTune().to(torch.device("cuda"))
    return Learner.from_checkpoint(r'F:\Stuff\Programming\experiments\vkr\training\RHUH v2 full + BRATS v2 test, 32\models\ft int+flip, contextnet-SegResNetDS+adan+dicefocal lr1e-03 OneCycleLR', model=model, cbs = ())