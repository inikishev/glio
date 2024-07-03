import torch
import torch.nn as nn


class LocalHistogramLayer1(nn.Module):
    def __init__(self, in_channels, out_channels):
        """Completely untested"""
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.bin_centers = nn.Parameter(torch.randn(out_channels, in_channels, requires_grad = True), True)
        self.bin_widths = nn.Parameter(torch.randn(out_channels, in_channels, requires_grad = True), True)

    def forward(self, x:torch.Tensor):
        # Create an empty tensor to store the histogram
        hist = torch.zeros(x.size(0), self.out_channels, x.size(2), x.size(3)).to(x.device)

        # Compute the histogram
        for i in range(self.out_channels):
            for j in range(self.in_channels):
                # Calculate the value for each channel of the histogram
                # using the Gaussian function
                # and add it to the corresponding position in the hist tensor
                hist[:, i, :, :] += torch.exp(-((x[:, j, :, :] - self.bin_centers[i, j])**2) / (2 * self.bin_widths[i, j]**2))

        return x

class LocalHistogramLayer2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, sigma=1.0):
        """Completely untested"""
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sigma = sigma

        # Initialize the bin centers and widths
        self.bin_centers = nn.Parameter(torch.randn(out_channels, in_channels), True)
        self.bin_widths = nn.Parameter(torch.ones(out_channels, in_channels), True)

    def forward(self, x: torch.Tensor):
        # Compute the distance between each bin center and each pixel
        distances = torch.sum((x.unsqueeze(1) - self.bin_centers.unsqueeze(0))**2, dim=2)
        # The distances variable calculates the squared Euclidean distance
        # between each bin center and each pixel in the input tensor x.
        # Compute the Gaussian RBF
        rbf = torch.exp(-distances / (2 * self.sigma**2))
        # The rbf variable applies the Gaussian RBF formula to the
        # computed distances, using a parameter sigma for the standard deviation.
        # Compute the histogram
        histogram = torch.sum(rbf.unsqueeze(2) * x.unsqueeze(1), dim=0)
        # The histogram variable calculates the weighted sum of the input tensor x
        # using the RBF values as weights.
        # TODO try not summing the histogram
        return histogram


class LocalHistogramLayer3(nn.Module):
    def __init__(self, in_channels, out_channels, bin_centers, bin_widths, kernel_size):
        """Completely untested"""
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.bin_centers = nn.Parameter(torch.FloatTensor(bin_centers), True)
        self.bin_widths = nn.Parameter(torch.FloatTensor(bin_widths), True)

        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_channels, out_channels * len(self.bin_centers), kernel_size, bias=False)

    def forward(self, x):
        # Apply convolutional operation to input tensor
        x = self.conv(x)
        # Apply ReLU activation function to the output of the convolutional operation
        x = torch.nn.functional.relu(x)

        # Reshape the tensor to have the desired dimensions
        x = x.view(-1, self.out_channels, len(self.bin_centers), x.size(-2), x.size(-1))

        # Calculate the histogram by applying the Gaussian kernel
        hist = torch.exp(-((x.unsqueeze(2) - self.bin_centers.view(1, 1, -1, 1, 1))**2) / (2 * self.bin_widths.view(1, 1, -1, 1, 1)**2))
        # Sum the values along the last two dimensions to obtain the histogram
        hist = hist.sum(dim=-1).sum(dim=-1)

        # Normalize the histogram by dividing each element by the sum of all elements in that row
        hist = hist / hist.sum(dim=-1).unsqueeze(-1)

        return hist


class Linear2D(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.weight = nn.Parameter(torch.normal(out_size, *in_size), True)
        self.bias = nn.Parameter(torch.randn(out_size, *in_size), True)
    def forward(self, x):
        return x @ self.weight + self.bias

if __name__ == "__main__":
    weight = torch.randn(8, 12, 16)
    batch = torch.randn(8, 16, 12)
    print((weight @ batch).shape)