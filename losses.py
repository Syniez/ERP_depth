import torch.nn as nn
import numpy as np
import torch



class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.name = 'MSE'
    
    def forward(self, src, dst, mask):
        diff = abs(src - dst)
        valid_pixel = torch.sum(mask, dim=[1, 2, 3], keepdim=True).float()

        masked_err = diff * mask.float()        
        return torch.mean(torch.sum(masked_err, dim=[1, 2, 3], keepdim=True) / valid_pixel)


class AngleLoss(nn.Module):
    def __init__(self):
        super(AngleLoss, self).__init__()
        self.name = 'Angle'
    
    def forward(self, src, dst):
        err = abs(src - dst)**2
        return err