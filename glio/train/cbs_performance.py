import torch
from ..design.EventModel import CBContext
from .Learner import Learner

class PerformanceTweaks(CBContext):
    def __init__(self, cudnn_bench, onednn_fusion=True, detect_anomaly=False, checknan = False, autograd_profiler = False, emit_nvtx=False, gradcheck=False, gradgradcheck=False):
        """Insane speed ups.

        Args:
            cudnn_bench (_type_): torch.backends.cudnn.benchmark
            onednn_fusion (bool, optional): torch.jit.enable_onednn_fusion. Defaults to True.
            detect_anomaly (bool, optional): torch.autograd.set_detect_anomaly. Defaults to False.
            checknan (bool, optional): torch.autograd.set_detect_anomaly. Defaults to False.
            autograd_profiler (bool, optional): torch.autograd.profiler.profile. Defaults to False.
            emit_nvtx (bool, optional): torch.autograd.profiler.emit_nvtx. Defaults to False.
            gradcheck (bool, optional): idk. Defaults to False.
            gradgradcheck (bool, optional): idk. Defaults to False.
        """
        super().__init__()
        self.cudnn_bench = cudnn_bench
        self.onednn_fusion = onednn_fusion
        self.detect_anomaly = detect_anomaly
        self.checknan = checknan
        self.autograd_profiler = autograd_profiler
        self.emit_nvtx = emit_nvtx
        self.gradcheck = gradcheck # TODO
        self.gradgradcheck = gradgradcheck # TODO

    def enter(self, learner: "Learner"):
        if self.cudnn_bench is not None: torch.backends.cudnn.benchmark = self.cudnn_bench
        if self.onednn_fusion is not None: torch.jit.enable_onednn_fusion(self.onednn_fusion)
        if self.detect_anomaly is not None: torch.autograd.set_detect_anomaly(self.detect_anomaly, self.checknan) # type:ignore
        if self.autograd_profiler is not None: torch.autograd.profiler.profile(self.autograd_profiler) # type:ignore
        if self.emit_nvtx is not None: torch.autograd.profiler.emit_nvtx(self.emit_nvtx) # type:ignore