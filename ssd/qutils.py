import torch


def get_delta(tensor):
    n = tensor[0].nelement()
    delta = .7 * tensor.norm(1, 3).sum(2).sum(1).div(n)
    return torch.repeat_interleave(delta, n).view(tensor.size())


def get_alpha(tensor, delta):
    n = tensor[0].nelement()
    x = torch.where(torch.abs(tensor) < delta,
                    torch.zeros_like(tensor),
                    tensor.sign())
    count = torch.abs(x).sum(1).sum(1).sum(1)
    abssum = (x*tensor).sum(1).sum(1).sum(1)
    alpha = abssum / count
    return torch.repeat_interleave(alpha, n).view(tensor.size())


def to_ternary(tensor, delta=None, alpha=None):
    if delta is None:
        delta = get_delta(tensor)
    if alpha is None:
        alpha = get_alpha(tensor, delta)

    x = torch.where(torch.abs(tensor) < delta,
                    torch.zeros_like(tensor),
                    tensor.sign())

    return x*alpha
