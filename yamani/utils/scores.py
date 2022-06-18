import torch


def calc_accuracy(labels, preds):
    """Calculates model accuracy

    Arguments:
        mdl {nn.model} -- nn model
        X {torch.Tensor} -- input data
        Y {torch.Tensor} -- labels/target values

    Returns:
        [torch.Tensor] -- accuracy
    """
    n = len(preds.squeeze())
    acc = (preds == labels).sum(dtype=torch.float32)/n
    return acc
