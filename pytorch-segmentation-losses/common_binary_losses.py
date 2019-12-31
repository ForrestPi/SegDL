import torch
import torch.nn as nn
import torch.nn.functional as F
from lovasz_losses_binary import lovasz_hinge
'''
These are some implements of binary segmentation loss
'''

class BCELoss2d(nn.Module):

    def __init__(self,weight=None,size_average=True):
        
        super(BCELoss2d,self).__init__()
        self.criterion = nn.BCELoss(weight,size_average)
    
    def forward(self,inputs,targets):
        probs = F.sigmoid(inputs)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        loss = self.criterion(probs_flat,targets_flat)
        
        return loss

class WeightBinaryCrossEntropy(nn.Module):
    
    def __init__(self, size_average=True):
        
        super(WeightBinaryCrossEntropy,self).__init__()
        self.size_average = size_average
    
    def forward(self,inputs,targets):
        
        mask = targets.float()
        num_positive = torch.sum((targets==1)).float()
        num_negative = torch.sum((targets==0)).float()

        mask[mask == 1] = 1.0 * num_negative / (num_positive + num_negative)
        mask[mask == 0] = 1.0 * num_positive / (num_positive + num_negative)

        loss = F.binary_cross_entropy(inputs,targets,weight=mask)
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
        
class FocalLoss(nn.Module):

    def __init__(self, alpha, gamma, OHEM_percent):
        super(FocalLoss,self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.OHEM_percent = OHEM_percent
    
    def forward(self,inputs,targets):

        inputs = inputs.contiguous().view(-1) 
        targets = targets.contiguous().view(-1)
        max_val = (-inputs).clamp(min=0)

        loss = inputs * (1 - targets) + max_val + ((-max_val).exp() + (-inputs - max_val).exp()).log()
       
        # This formula gives us the log sigmoid of 1-p if y is 0 and of p if y is 1
        invprobs = F.logsigmoid(-inputs * (targets * 2 - 1))
        focal_loss = self.alpha * (invprobs * self.gamma).exp() * loss
        # Online Hard Example Mining: top x% losses (pixel-wise). Refer to http://www.robots.ox.ac.uk/~tvg/publications/2017/0026.pdf
        OHEM, _ = focal_loss.topk(k=int(self.OHEM_percent * [*focal_loss.shape][0]))
        
        return OHEM.mean()

class LovaszSoftmax(nn.Module):
    def __init__(self, classes='present', per_image=False, ignore=-1):
        super(LovaszSoftmax, self).__init__()
        self.smooth = classes
        self.per_image = per_image
        self.ignore_index = ignore
    
    def forward(self, output, target):
        logits = F.softmax(output, dim=1)
        loss = lovasz_hinge(logits, target, ignore=self.ignore_index)
        return loss