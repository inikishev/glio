import torch
def iou(y:torch.Tensor, yhat:torch.Tensor):
    """
    y: ground truth in one hot format, must be of BC* shape
    yhat: prediction in one hot format, must be of BC* shape

    returns: vector of len C with iou per each channel
    """
    y = y.to(torch.bool)
    yhat = yhat.to(torch.bool)
    intersection = (y & yhat).sum((0, *list(range(2, y.ndim))))
    union = (y | yhat).sum((0, *list(range(2, y.ndim))))
    return intersection / union