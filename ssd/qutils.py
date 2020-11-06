import torch


def to_ternary(tensor, delta=None, alpha=None):
    n = tensor[0].nelement()

    if delta is None:
        delta = .7 * tensor.norm(1, 3).sum(2).sum(1).div(n)
        delta = torch.repeat_interleave(delta, n).view(tensor.size())

    x = torch.where(torch.abs(tensor) < delta,
                    torch.zeros_like(tensor),
                    tensor.sign())

    if alpha is None:
        count = torch.abs(x).sum(1).sum(1).sum(1)
        abssum = (x*tensor).sum(1).sum(1).sum(1)
        alpha = abssum / count
        alpha = torch.repeat_interleave(alpha, n).view(tensor.size())

    return x*alpha
