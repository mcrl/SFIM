import numpy as np
import os
from torch.utils.data import Dataset
import torch
from utils.image_utils import is_png_file, load_img, load_npy, is_target_file
import torch.nn.functional as F
import random
from PIL import Image
import torchvision.transforms.functional as TF
from natsort import natsorted

class DataLoaderTrain(Dataset):
    def __init__(self, opt, rgb_dir, data_format, patch_size):
        super(DataLoaderTrain, self).__init__()
        self.opt = opt
        self.ps = patch_size
        self.data_format = data_format
        self.ir = 0
        self.opt = opt
        
        gt_dir = 'GT' 
        input_dir = 'input'

        isTargetFile = is_target_file(data_format)
        
        clean_files = sorted(os.listdir(os.path.join(rgb_dir, gt_dir)))
        noisy_files = sorted(os.listdir(os.path.join(rgb_dir, input_dir)))
        self.clean_filenames = [os.path.join(rgb_dir, gt_dir, x) for x in clean_files if isTargetFile(x)]
        self.noisy_filenames = [os.path.join(rgb_dir, input_dir, x) for x in noisy_files if isTargetFile(x)]

        self.tar_size = len(self.clean_filenames)

    def return_size(self):
        if self.data_format != 0:
            clean = torch.from_numpy(np.float32(load_img(self.opt,self.clean_filenames[0])))
        else:
            clean = torch.from_numpy(np.float32(np.load(self.clean_filenames[0])))
        
        clean = clean.permute(self.opt.perm_args) # to CHW
        return clean.shape[0], clean.shape[1], clean.shape[2]

    def __len__(self):
        return self.tar_size
        
    def __getitem__(self, index):
        tar_index = index

        if self.data_format != 0:
            clean = torch.from_numpy(np.float32(load_img(self.opt, self.clean_filenames[tar_index])))
            noisy = torch.from_numpy(np.float32(load_img(self.opt, self.noisy_filenames[tar_index])))
        else:
            clean = torch.from_numpy(np.float32(load_npy(self.opt, self.clean_filenames[tar_index])))
            noisy = torch.from_numpy(np.float32(load_npy(self.opt, self.noisy_filenames[tar_index])))

        clean = clean.permute(self.opt.perm_args)
        noisy = noisy.permute(self.opt.perm_args)

        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]
        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]

        # Crop Input and Target
        H, W = clean.shape[1], clean.shape[2]
        row = np.random.randint(0, H - self.ps) if H-self.ps != 0 else 0
        col = np.random.randint(0, W - self.ps) if W-self.ps != 0 else 0
        clean = clean[:, row:row + self.opt.patch_size, col:col + self.opt.patch_size]
        noisy = noisy[:, row:row + self.ps, col:col + self.ps]

        return clean, noisy, clean_filename, noisy_filename

##################################################################################################
class DataLoaderVal(Dataset):
    def __init__(self, opt, rgb_dir, data_format):
        super(DataLoaderVal, self).__init__()

        self.data_format = data_format
        self.opt = opt

        gt_dir = 'GT'
        input_dir = 'input'

        isTargetFile = is_target_file(data_format)

        clean_files = sorted(os.listdir(os.path.join(rgb_dir, gt_dir)))
        noisy_files = sorted(os.listdir(os.path.join(rgb_dir, input_dir)))
        self.clean_filenames = [os.path.join(rgb_dir, gt_dir, x) for x in clean_files if isTargetFile(x)]
        self.noisy_filenames = [os.path.join(rgb_dir, input_dir, x) for x in noisy_files if isTargetFile(x)]           

        self.tar_size = len(self.clean_filenames)  

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index   = index % self.tar_size
        clean = None
        noisy = None
        if self.data_format != 0:
            clean = torch.from_numpy(np.float32(load_img(self.opt, self.clean_filenames[tar_index])))
            noisy = torch.from_numpy(np.float32(load_img(self.opt, self.noisy_filenames[tar_index])))
        else:
            clean = torch.from_numpy(np.float32(load_npy(self.opt, self.clean_filenames[tar_index])))
            noisy = torch.from_numpy(np.float32(load_npy(self.opt, self.noisy_filenames[tar_index])))
                
        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]
        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]

        clean = clean.permute(self.opt.perm_args)#2,0,1)
        noisy = noisy.permute(self.opt.perm_args)#2,0,1)

        return clean, noisy, clean_filename, noisy_filename



class DataLoaderTest(Dataset):
    def __init__(self, opt, rgb_dir, data_format):
        super(DataLoaderTest, self).__init__()

        self.data_format = data_format
        self.opt = opt

        gt_dir = 'GT'
        input_dir = 'input'

        isTargetFile = is_target_file(data_format)

        clean_files = sorted(os.listdir(os.path.join(rgb_dir, gt_dir)))
        noisy_files = sorted(os.listdir(os.path.join(rgb_dir, input_dir)))
        self.clean_filenames = [os.path.join(rgb_dir, gt_dir, x) for x in clean_files if isTargetFile(x)]
        self.noisy_filenames = [os.path.join(rgb_dir, input_dir, x) for x in noisy_files if isTargetFile(x)]           

        self.tar_size = len(self.clean_filenames)  

    def return_size(self):
        if self.data_format != 0:
            clean = torch.from_numpy(np.float32(load_img(self.opt, self.clean_filenames[0])))
        else:
            clean = torch.from_numpy(np.float32(np.load(self.clean_filenames[0])))
        
        clean = clean.permute([2,0,1]) # to CHW
        return clean.shape[0], clean.shape[1], clean.shape[2]

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index   = index % self.tar_size
        clean = None
        noisy = None
        if self.data_format != 0:
            clean = torch.from_numpy(np.float32(load_img(self.opt, self.clean_filenames[tar_index])))
            noisy = torch.from_numpy(np.float32(load_img(self.opt, self.noisy_filenames[tar_index])))
        else:
            clean = torch.from_numpy(np.float32(load_npy(self.opt, self.clean_filenames[tar_index])))
            noisy = torch.from_numpy(np.float32(load_npy(self.opt, self.noisy_filenames[tar_index])))
                
        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]
        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]

        
        clean = clean.permute([2,0,1])#self.opt.perm_args)#2,0,1)
        noisy = noisy.permute([2,0,1])#self.opt.perm_args)#2,0,1)

        return clean, noisy, clean_filename, noisy_filename
