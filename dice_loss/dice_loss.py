import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
class MAELoss(nn.Module):
    def __init__(self):
        super(MAELoss, self).__init__()
    def forward(self, pred, target):
        mae_loss = torch.mean(torch.abs(pred - target))
        return mae_loss