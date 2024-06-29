import torch
def dice(y:torch.Tensor, yhat:torch.Tensor):
    """
    y: ground truth in one hot format, must be of BC* shape
    yhat: prediction in one hot format, must be of BC* shape

    returns: vector of len C with dice per each channel
    """
    y = y.to(torch.bool)
    yhat = yhat.to(torch.bool)
    intersection = (y & yhat).sum((0, *list(range(2, y.ndim))))
    sum_ = y.sum((0, *list(range(2, y.ndim)))) + yhat.sum((0, *list(range(2, y.ndim))))
    return (2*intersection) / sum_