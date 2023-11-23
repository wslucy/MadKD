from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

class CSKDLoss(nn.Module):
    """
    CSKD by Wslucy:
    T target clip to the top2(clip by delta), S target clip by lambda. 
    lambda==delta
    """
    def  __init__(self, alpha = 1, beta = 8, temperature = 4):
        super(STRKDLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature

    def forward(self, y_s, y_t, target, epoch):
        # correct strict y_t, y_s
        y_t, y_s = correct_logits(y_t, y_s)
        # KL loss
        loss_kd = kd_loss(y_s, y_t, self.temperature)
        return loss_kd 

def correct_logits(y_t, y_s, k=1, n=2):
    y_t, y_s = y_t.clone(), y_s.clone()
    # get topk indics
    max_val, _ = torch.topk(y_t, k=2, dim=1)
    # top2 val top2 class
    top2_val = max_val[:,-1].unsqueeze(1)
    # top val  target class
    top1_val = max_val[:,k-1].unsqueeze(1)
    # cal topk
    y_t_top_ind = torch.where(y_t >= top1_val, 1, 0)
    # get top1 and top2 matrix 
    top1_val_matrix = top1_val * y_t_top_ind
    top2_val_matrix = top2_val * y_t_top_ind
    # cal delta
    delta_matrix = top1_val_matrix - top2_val_matrix
    y_t = y_t - delta_matrix
    y_s = y_s - delta_matrix

    return y_t, y_s

def kd_loss(y_s, y_t, temperature):
    p_s = F.log_softmax(y_s/temperature, dim=1)
    p_t = F.softmax(y_t/temperature, dim=1)
    loss = nn.KLDivLoss(reduction='batchmean')(p_s, p_t) * (temperature**2)
    return loss
