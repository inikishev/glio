def conv_outsize(in_size:tuple,kernel_size, stride = 1, padding = 0,output_padding = 0, dilation=1):
    """    Convolve 2D input with a kernel.

    This function calculates the output size after applying 2D convolution
    operation on the input size with the given kernel size, stride, padding,
    output padding, and dilation.

    Args:
        in_size (tuple): The size of the input tensor.
        kernel_size: The size of the convolutional kernel.
        stride (int): The stride of the convolution operation. Default is 1.
        padding (int): The zero-padding added to both sides of the input. Default is 0.
        output_padding (int): Additional size added to one side of the output shape. Default is 0.
        dilation (int): The spacing between kernel elements. Default is 1.

    Returns:
        list: The size of the output tensor after convolution.
    """
    if isinstance(in_size, int): in_size = (in_size,)
    if isinstance(kernel_size, int): kernel_size = [kernel_size]*len(in_size)
    if isinstance(stride, int): stride = [stride]*len(in_size)
    if isinstance(padding, int): padding = [padding]*len(in_size)
    if isinstance(output_padding, int): output_padding = [output_padding]*len(in_size)
    if isinstance(dilation, int): dilation = [dilation]*len(in_size)
    out_size = [int((in_size[i] + 2 * padding[i] - dilation[i] * (kernel_size[i] - 1) - 1) / stride[i] + 1) for i in range(len(in_size))]
    print(out_size)
    return out_size

def convtranpose_outsize(in_size:tuple,kernel_size, stride = 1, padding = 0, output_padding = 0, dilation=(1,1)):
    """    Calculate the output size after a 2D transposed convolution operation.

    This function calculates the output size based on the input size, kernel
    size, stride, padding, output padding, and dilation for a 2D transposed
    convolution operation.

    Args:
        in_size (tuple): The input size as a tuple of integers.
        kernel_size: The size of the convolutional kernel.
        stride (int): The stride of the convolution operation. Default is 1.
        padding (int): The amount of zero-padding added to the input. Default is 0.
        output_padding (int): The additional size added to the output shape. Default is 0.
        dilation (tuple): The dilation rate for each dimension. Default is (1, 1).

    Returns:
        list: A list containing the calculated output size for each dimension.
    """
    if isinstance(in_size, int): in_size = (in_size,)
    if isinstance(kernel_size, int): kernel_size = [kernel_size]*len(in_size)
    if isinstance(stride, int): stride = [stride]*len(in_size)
    if isinstance(padding, int): padding = [padding]*len(in_size)
    if isinstance(output_padding, int): output_padding = [output_padding]*len(in_size)
    if isinstance(dilation, int): dilation = [dilation]*len(in_size)
    out_size = [int((in_size[i]-1)*stride[i] - 2*padding[i] + dilation[i]*(kernel_size[i]-1) + output_padding[i] + 1) for i in range(len(in_size))]
    print(out_size)
    return out_size

