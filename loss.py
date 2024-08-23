import torch
import torch.nn as nn
import torch.nn.functional as F

def dice_loss(predicted, target):
    smooth = 1e-5
    intersection = torch.sum(predicted * target)
    union = torch.sum(predicted) + torch.sum(target)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1 - dice

def jaccard_loss(predicted, target):
    smooth = 1e-5
    intersection = torch.sum(predicted * target)
    union = torch.sum(predicted) + torch.sum(target) - intersection
    jaccard = (intersection + smooth) / (union + smooth)
    return 1 - jaccard

class MixedLoss(nn.Module):
    def __init__(self, weight_nll=1.0, weight_dice=1.0, weight_jaccard=1.0):
        super(MixedLoss, self).__init__()
        self.weight_nll = weight_nll
        self.weight_dice = weight_dice
        self.weight_jaccard = weight_jaccard
        self.mse = nn.MSELoss()

    def forward(self, predicted, target):
        nll = self.mse(predicted, target)
        dice = dice_loss(predicted, target)
        jaccard = jaccard_loss(predicted, target)
        loss = self.weight_nll * nll + self.weight_dice * dice + self.weight_jaccard * jaccard
        return loss