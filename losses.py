import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ignite.metrics import SSIM
from torchmetrics.image import StructuralSimilarityIndexMeasure

class CharbonnierLoss(nn.Module):

    def __init__(self, eps=1e-4): # eps: 1e-3 -> 1e-4 (240215 KYUSU)
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):#, mask=None):
        diff = x - y
        # loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps))) # WRONG LOSS FUNCTION: MODIFY TO BELOW (KYUSU 240215)
        # loss = torch.sqrt( torch.sum( (diff * diff) + (self.eps*self.eps) ) )
        loss = torch.sqrt( torch.mean( (diff * diff) + (self.eps*self.eps) ) )
        return loss

class SSIM_Loss(torch.nn.Module):

    def __init__(self, data_range=1.0):
        super(SSIM_Loss, self).__init__()
        self.ssim_metric = StructuralSimilarityIndexMeasure()#SSIM(data_range=data_range)
        # self.ssim_metric.attach(default_evaluator, 'ssim')

    def forward(self, _input, gt):
        # Calculate SSIM loss
        loss_ssim = 1 - self.ssim_metric(_input, gt)
        loss_ssim = torch.mean(loss_ssim)

        return loss_ssim

class FFT_loss(nn.Module):

    def __init__(self):
        super(FFT_loss, self).__init__()
        self.criterion = torch.nn.L1Loss()
    
    def forward(self, lq, gt):
        fft_lq = torch.fft.fftn(lq, dim=(-2, -1))
        fft_gt = torch.fft.fftn(gt, dim=(-2, -1))
        loss_fft = F.l1_loss(fft_lq, fft_gt, reduction="mean")
        return loss_fft

class FFT_abs_L1(nn.Module):

    def __init__(self):
        super(FFT_abs_L1, self).__init__()
        self.criterion = torch.nn.L1Loss()

    def forward(self, _input, gt):
        input_fft = abs( torch.fft.fft2(_input) )
        gt_fft = abs( torch.fft.fft2(gt) )
        loss_fft = self.criterion(input_fft, gt_fft)
        return loss_fft

class FFT_angle_L1(nn.Module):

    def __init__(self):
        super(FFT_angle_L1, self).__init__()
        self.criterion = torch.nn.L1Loss()

    def forward(self, _input, gt):
        input_fft = torch.angle( torch.fft.fft2(_input))  
        gt_fft = torch.angle( torch.fft.fft2(gt) ) 
        loss_fft = self.criterion(input_fft, gt_fft)
        return loss_fft

class FFTLoss_real(nn.Module):

    def __init__(self):
        super(FFTLoss_real, self).__init__()
        self.criterion = torch.nn.L1Loss()

    def forward(self, pred, target):
        pred_fft = torch.fft.fft2(pred, dim=(-2, -1))
        target_fft = torch.fft.fft2(target, dim=(-2, -1))
        loss_fft = self.criterion(pred_fft.real, target_fft.real)
        return loss_fft

class FFTLoss_imag(nn.Module):

    def __init__(self):
        super(FFTLoss_imag, self).__init__()
        self.criterion = torch.nn.L1Loss()

    def forward(self, pred, target):

        pred_fft = torch.fft.fft2(pred, dim=(-2, -1))
        target_fft = torch.fft.fft2(target, dim=(-2, -1))
        loss_fft = self.criterion(pred_fft.imag, target_fft.imag)
        return loss_fft        