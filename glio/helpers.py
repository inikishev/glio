# Автор - Никишев Иван Олегович группа 224-31

def cnn_output_size(in_size:tuple,kernel_size, stride = 1, padding = 0,output_padding = 0, dilation=1):
    "conv 2d"
    if isinstance(in_size, int): in_size = (in_size,)
    if isinstance(kernel_size, int): kernel_size = [kernel_size]*len(in_size)
    if isinstance(stride, int): stride = [stride]*len(in_size)
    if isinstance(padding, int): padding = [padding]*len(in_size)
    if isinstance(output_padding, int): output_padding = [output_padding]*len(in_size)
    if isinstance(dilation, int): dilation = [dilation]*len(in_size)
    out_size = [int((in_size[i] + 2 * padding[i] - dilation[i] * (kernel_size[i] - 1) - 1) / stride[i] + 1) for i in range(len(in_size))]
    print(out_size)
    return out_size

def tcnn_output_size(in_size:tuple,kernel_size, stride = 1, padding = 0, output_padding = 0, dilation=(1,1)):
    """conv transpose 2d"""
    if isinstance(in_size, int): in_size = (in_size,)
    if isinstance(kernel_size, int): kernel_size = [kernel_size]*len(in_size)
    if isinstance(stride, int): stride = [stride]*len(in_size)
    if isinstance(padding, int): padding = [padding]*len(in_size)
    if isinstance(output_padding, int): output_padding = [output_padding]*len(in_size)
    if isinstance(dilation, int): dilation = [dilation]*len(in_size)
    out_size = [int((in_size[i]-1)*stride[i] - 2*padding[i] + dilation[i]*(kernel_size[i]-1) + output_padding[i] + 1) for i in range(len(in_size))]
    print(out_size)
    return out_size

