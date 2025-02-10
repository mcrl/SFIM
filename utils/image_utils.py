import torch
import numpy as np
import pickle
import cv2
import imageio
import rawpy as rp
import os
from PIL import Image

import torch.nn.functional as F
from math import exp

#from skimage.color.adapt_rgb import demosaicing
from colour_demosaicing import demosaicing_CFA_Bayer_bilinear

class SSIM(object):
  ''' 
  modified from https://github.com/jorge-pessoa/pytorch-msssim
  '''
  def __init__(self, des="structural similarity index"):
    self.des = des 
  
  def __repr__(self):
    return "SSIM"
  
  def gaussian(self, w_size, sigma):
    gauss = torch.Tensor([exp(-(x - w_size//2)**2/float(2*sigma**2)) for x in range(w_size)])
    return gauss/gauss.sum()
  
  def create_window(self, w_size, channel=1):
    _1D_window = self.gaussian(w_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, w_size, w_size).contiguous()
    return window
  
  def __call__(self, y_pred, y_true, w_size=11, size_average=True, full=False):
    """ 
    args:
        y_true : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
        y_pred : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
        w_size : int, default 11
        size_average : boolean, default True
        full : boolean, default False
    return ssim, larger the better
    """
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if torch.max(y_pred) > 128:
      max_val = 255 
    else:
      max_val = 1 
    
    if torch.min(y_pred) < -0.5:
      min_val = -1
    else:
      min_val = 0 
    L = max_val - min_val
    
    padd = 0 
    (_, channel, height, width) = y_pred.size()
    window = self.create_window(w_size, channel=channel).to(y_pred.device)

    mu1 = F.conv2d(y_pred, window, padding=padd, groups=channel)
    mu2 = F.conv2d(y_true, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(y_pred * y_pred, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(y_true * y_true, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(y_pred * y_true, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
      ret = ssim_map.mean()
    else:
      ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
      return ret, cs
    return ret

def calc_metric(y_true, y_pred, size_average=True): 
    metric = SSIM()
    ssim = metric(y_pred, y_true, size_average=size_average)
    return ssim


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):

    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret

def tonemap(x, type='simple'):
    if type == 'mu_law':
        norm_x = x / x.max()
        mapped_x = np.log(1 + 10000 * norm_x) / np.log(1 + 10000)
    elif type == 'simple':
        mapped_x = x / (x + 0.25)
    elif type == 'same':
        mapped_x = x
    else:
        raise NotImplementedError('tone mapping type [{:s}] is not recognized.'.format(type))
    return mapped_x

def is_target_file(targetType):
    if targetType == 0:
        return is_numpy_file
    elif targetType == 1:
        return is_png_file
    elif targetType == 2:
        return is_image_file
    elif targetType == 3:
        return is_pkl_file

def is_numpy_file(filename):
    return any(filename.endswith(extension) for extension in [".npy"])

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".jpg"])

def is_png_file(filename):
    return any(filename.endswith(extension) for extension in [".png"])

def is_pkl_file(filename):
    return any(filename.endswith(extension) for extension in [".pkl"])

def load_pkl(filename_):
    with open(filename_, 'rb') as f:
        ret_dict = pickle.load(f)
    return ret_dict    

def save_dict(dict_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(dict_, f)    

def load_npy(opt, filepath):
    img = np.load(filepath)
    if opt.tonemap:
        img = tonemap(img)
    else:
        img = img/opt.max_pxl
        
    return img

def load_img(opt, filepath):
    img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
    # img = img.astype(np.float32)
    if opt.tonemap:
        img = tonemap(img)
    else:
        img = img/opt.max_pxl

    return img

def save_img(filepath, img):
    cv2.imwrite(filepath,cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def save_output(opt, tensor_to_save, epoch, res_dir, fnames, save_type=None, psnr=0.0, ssim=0.0):
    # print("Saving outputs")
    if type(psnr) is list: psnr = psnr[0]
    if type(ssim) is list: ssim = ssim[0]

    if opt.data_name == 'Feng-S' or opt.data_name == 'Zhou-S':
        save_3ch_tensor_png(opt, tensor_to_save, epoch, res_dir, fnames, save_type, psnr, ssim)
    elif opt.data_name == 'UDC_SIT':
        save_4ch_npy_png(opt, tensor_to_save, epoch, res_dir, fnames, save_type, psnr, ssim)
    else:
        ValueError("Unsupported dataset")

def save_3ch_tensor_png(opt, tensor_to_save, epoch, res_dir, fnames, save_type, psnr=0, ssim=0):
    fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
    if epoch >= 0:
        str_epoch = "Epoch_" + str(epoch)
    else:
        str_epoch = ""
    if not os.path.exists(os.path.join(res_dir,str_epoch)):
        os.makedirs(os.path.join(res_dir,str_epoch))        
    for i in range(tensor_to_save.shape[0]):
        npy = fn_tonumpy(tensor_to_save)
        img = (npy[i,:,:,:].squeeze()*255).astype(np.uint8) # opt.max_pxl
        img = Image.fromarray(img, mode='RGB')
        fname = fnames[i].split('.')[0]
        if save_type is None:
            output_name = f"{res_dir}{str_epoch}/{fname}_{psnr:.4f}_{ssim:.4f}.png"
            img.save(output_name)
        if save_type is not None:
            output_name = f"{res_dir}{str_epoch}/{fname}{save_type}_{psnr:.4f}_{ssim:.4f}.png"
            img.save(output_name)

def save_4ch_npy_png(opt, tensor_to_save, epoch, res_dir, fnames, save_type, psnr=0, ssim=0):
    fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 1, 3, 2)
    data  = rp.imread(opt.source_dir)
    if epoch >= 0:
        str_epoch = "Epoch_" + str(epoch)
    else:
        str_epoch = ""
    if not os.path.exists(os.path.join(res_dir,str_epoch)):
        os.makedirs(os.path.join(res_dir,str_epoch), exist_ok=True)
    for idx in range(tensor_to_save.shape[0]):
        npy = fn_tonumpy(tensor_to_save)  
        npy = (npy[idx,:,:,:].squeeze()*opt.max_pxl)#255) # opt.max_pxl
        GR = data.raw_image[0::2,0::2]
        R = data.raw_image[0::2,1::2]
        B = data.raw_image[1::2,0::2]
        GB = data.raw_image[1::2,1::2]
        GB[:,:] = 0
        B[:,:] = 0
        R[:,:] = 0
        GR[:,:] = 0
        for k in range(4):
            for i in range(npy.shape[1]):
                for j in range(npy.shape[2]):
                    if(k==0):
                        GR[i][j] = npy[k][i][j]
                    elif(k==1):
                        R[i][j] = npy[k][i][j]
                    elif(k==2):
                        B[i][j] = npy[k][i][j]
                    elif(k==3):    
                        GB[i][j] = npy[k][i][j]
        newData = data.postprocess()
        start = (0,464)# (448 , 0) # 1792 1280 ->   3584, 2560
        end =  (3584,3024)
        newData = newData[start[0]:end[0], start[1]:end[1]]        
        fname = fnames[idx].split('.')[0]
        if save_type is None:
            output_name = f"{res_dir}{str_epoch}/{fname}_{psnr:.4f}_{ssim:.4f}.png"
        if save_type is not None:
            output_name = f"{res_dir}{str_epoch}/{fname}{save_type}_{psnr:.4f}_{ssim:.4f}.png"
        imageio.imsave(output_name, newData)


def myPSNR(tar_img, prd_img):
    imdff = torch.clamp(prd_img,0,1) - torch.clamp(tar_img,0,1)
    rmse = (imdff**2).mean().sqrt()
    ps = 20*torch.log10(1/rmse)
    return ps, rmse

def myRMSE(tar_img, prd_img):
    imdff = (torch.clamp(prd_img,0,1) * 255).round() - (torch.clamp(tar_img,0,1) * 255).round()
    rmse = (imdff**2).mean().sqrt().item()
    return rmse

def batch_PSNR(img1, img2):
    import math
    PSNR, RMSE = [], []
    for im1, im2 in zip(img1, img2):
        #psnr, rmse = myPSNR(im1, im2)
        rmse = myRMSE(im1, im2)
        if rmse != 0:
            PSNR.append(20*math.log10(255/rmse))
            RMSE.append(rmse)
    return PSNR, RMSE

def batch_SSIM(img1, img2):
    SSIM = []
    for im1, im2 in zip(img1, img2):
        SSIM.append(ssim(im1, im2))
    return SSIM

def each_PSNR(img1, img2):
    PSNR, RMSE = [], []
    for im1, im2 in zip(img1, img2):
        #psnr, rmse = myPSNR(im1, im2)
        rmse = myRMSE(im1, im2)
        if rmse != 0:
            PSNR.append(20*torch.log10(255/rmse))
            RMSE.append(rmse)
    return PSNR

def each_SSIM(img1, img2):
    SSIM = []
    for im1, im2 in zip(img1, img2):
        SSIM.append(ssim(im1, im2))
    return SSIM