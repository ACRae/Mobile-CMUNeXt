import torch
from torch import nn
import torch.nn.functional as f


__all__ = ["BCEDiceLoss"]


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = f.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = input * target
        dice = (2.0 * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice


def compute_kl_loss(p, q):
    p_loss = f.kl_div(f.log_softmax(p, dim=-1), f.softmax(q, dim=-1), reduction="none")
    q_loss = f.kl_div(f.log_softmax(q, dim=-1), f.softmax(p, dim=-1), reduction="none")

    p_loss = p_loss.mean()
    q_loss = q_loss.mean()

    return (p_loss + q_loss) / 2
