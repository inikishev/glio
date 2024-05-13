from glio.train2 import *
from monai.networks.nets import VNet # type:ignore
from schedulefree import AdamWScheduleFree
MODEL = VNet(2, 4,4)
OPT = AdamWScheduleFree(MODEL.parameters(), lr=1e-2, eps=1e-6)
path = r"F:\Stuff\Programming\AI\glio_diff\glio\models\glio postop segm\refining tandem v1\1. v1 VNet"
l = Learner.from_checkpoint(path, MODEL, [Accelerate("no"),], optimizer=OPT)
l.fit(0, None, None)