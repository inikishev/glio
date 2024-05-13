from glio.train2 import *
from monai.networks.nets import SegResNetDS # type:ignore
from schedulefree import AdamWScheduleFree
MODEL = SegResNetDS(2, in_channels=12, out_channels=4, init_filters=24)
OPT = AdamWScheduleFree(MODEL.parameters(), lr=1e-3, eps=1e-6)
path = r"F:\Stuff\Programming\AI\glio_diff\glio\models\glio postop segm\refining tandem v1\2. v1 SegResNetDS"
l = Learner.from_checkpoint(path, MODEL, [Accelerate("no"),], optimizer=OPT)
l.fit(0, None, None)