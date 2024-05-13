"""Свёртка обучаемой матрицей"""
import torch

def apply_learned_conv(x, filters):
    return torch.stack([torch.nn.functional.conv2d(image, filters[i]) for i, image in enumerate(x)]) # pylint: disable=E1102

class LMConv(torch.nn.Module):
    def __init__(self, mat_size, in_channels, bias = True, mode = "in", stride = 1, padding = 0, dilation = 1, init = torch.nn.init.kaiming_normal_):
        """
        Uses input as kernel to convolve a learnable matrix of size `mat_size` which needs to be `(H, W)` for 2d input.

        If `mode` is `in`, input channels are used to convolve the same amount of channels in the weight, producing a 1 channel output.

        If `mode` is `out`, input channels are used to convolve 1-channeled weight, producing the same amount of channels as input.

        Supports 1d, 2d or 3d input size + channel dimension, which is inferred from `mat_size`.
        """
        super().__init__()

        if isinstance(mat_size, int): mat_size = (mat_size,)
        self.weight = torch.nn.Parameter(torch.zeros(1, in_channels, *mat_size), True) if mode == 'in' else torch.nn.Parameter(torch.randn(1, 1, *mat_size), True)
        """Matrix of size `(1, in_channels, H, W)`, where `1` is the batch dimension, as convolution is applied to each element of batch separately."""
        init(self.weight)

        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(in_channels if mode == "out" else 1), True)
            self.has_bias = True
        else: self.has_bias = False
        self.mode = 1 if mode == "in" else 2

        self.ndim = len(mat_size)
        if self.ndim == 1 : conv = torch.nn.functional.conv1d
        elif self.ndim == 2: conv = torch.nn.functional.conv2d
        elif self.ndim == 3: conv = torch.nn.functional.conv3d
        else: raise NotImplementedError(f"ReverseConv only supports 1d, 2d and 3d convolutions, not {self.ndim}d")
        self.batched_rev_conv = torch.func.vmap(lambda x: conv(input=self.weight, weight = x, bias=self.bias if self.has_bias else None, stride=stride, padding=padding, dilation=dilation)) # pylint: disable=E1102 # type: ignore

    def forward(self, x):
        # weight, x, must be (out_channels, in_channels, H, W)
        # it is (batch, channels, H, W)
        # input, self.weights, must be (minibatch, in_channels, H, W)
        return self.batched_rev_conv(x.unsqueeze(self.mode)).squeeze(1)

        # if self.has_bias: return torch.stack([self.conv(self.weight, batch_el, bias = self.bias) for batch_el in x.unsqueeze(self.mode)]).squeeze(1)
        # return torch.stack([self.conv(self.weight, batch_el) for batch_el in x.unsqueeze(self.mode)]).squeeze(1)

class LMConvTranspose(torch.nn.Module):
    def __init__(self, mat_size, in_channels, bias = True, mode = "in", stride = 1, padding = 0, dilation = 1, init = torch.nn.init.kaiming_normal_):
        """
        Uses input as kernel to convolve a learnable matrix of size `mat_size` which needs to be `(H, W)` for 2d input.

        If `mode` is `in`, input channels are used to convolve the same amount of channels in the weight, producing a 1 channel output.

        If `mode` is `out`, input channels are used to convolve 1-channeled weight, producing the same amount of channels as input.

        Supports 1d, 2d or 3d input size + channel dimension, which is inferred from `mat_size`.
        """
        super().__init__()

        if isinstance(mat_size, int): mat_size = (mat_size,)
        self.weight = torch.nn.Parameter(torch.zeros(1, in_channels, *mat_size), True) if mode == 'in' else torch.nn.Parameter(torch.randn(1, 1, *mat_size), True)
        """Matrix of size `(1, in_channels, H, W)`, where `1` is the batch dimension, as convolution is applied to each element of batch separately."""
        init(self.weight)

        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(in_channels if mode == "out" else 1), True)
            self.has_bias = True
        else: self.has_bias = False
        self.mode = 2 if mode == "in" else 1

        self.ndim = len(mat_size)
        if self.ndim == 1 : conv = torch.nn.functional.conv_transpose1d
        elif self.ndim == 2: conv = torch.nn.functional.conv_transpose2d
        elif self.ndim == 3: conv = torch.nn.functional.conv_transpose3d
        else: raise NotImplementedError(f"ReverseConvTranspose only supports 1d, 2d and 3d convolutions, not {self.ndim}d")
        self.batched_rev_conv = torch.func.vmap(lambda x: conv(input=self.weight, weight = x, bias=self.bias if self.has_bias else None, stride=stride, padding=padding, dilation=dilation)) # pylint: disable=E1102 # type: ignore

    def forward(self, x):
        # weight, x, must be (in_channels, out_channels, H, W)
        # it is (batch, channels, H, W)
        # input, self.weights, must be (minibatch, in_channels, H, W)
        return self.batched_rev_conv(x.unsqueeze(self.mode)).squeeze(1)

        # if self.has_bias: return torch.stack([self.conv(self.weight, batch_el, bias = self.bias) for batch_el in x.unsqueeze(self.mode)]).squeeze(1)
        # return torch.stack([self.conv(self.weight, batch_el) for batch_el in x.unsqueeze(self.mode)]).squeeze(1)



if __name__ == "__main__":
    rc_in = LMConv((64, 64), 3, bias = True, mode = "in")
    batch = torch.randn(16, 3, 28, 28)
    print(rc_in(batch).shape)

    rc_out = LMConv((64, 64), 3, bias = True, mode = "out")
    print(rc_out(batch).shape)
