import random
import numpy as np
import torch
import os
import datetime
import utils
from torch.optim.lr_scheduler import StepLR
from warmup_scheduler import GradualWarmupScheduler
import torch.distributed as dist

def automatic_opt_setting(opt):

    if opt.data_name == 'UDC-SIT':
        opt.num_ch = 4
        opt.perm_args = [2,0,1] 
        if opt.data_format == 0: # npy
            opt.max_pxl = 1023.0
        elif opt.data_format == 1 or opt.data_format == 2: # png or jpg
            opt.max_pxl = 255.0

    elif opt.data_name == 'Feng-S' or opt.data_name == 'SYNTH':
        opt.num_ch = 3
        opt.perm_args = [2,0,1]
        opt.tonemap = True
        opt.max_pxl = 500.0
    
    elif opt.data_name == 'Zhou-S' or opt.data_name == 'P-OLED' or opt.data_name == 'T-OLED':
        opt.num_ch = 3
        opt.perm_args = [2,0,1]
        opt.tonemap = False
        opt.data_format = 1 # png
        opt.max_pxl = 255.0

    return opt


def set_seeds(num):
    random.seed(num)
    np.random.seed(num)
    torch.manual_seed(num)
    torch.cuda.manual_seed_all(num)


def set_dirs(opt):

    if opt.save_name == '':
        opt.log_dir = f"{opt.log_dir}/{opt.data_name}_{opt.arch}"
        opt.log_dir += f"_{opt.model_construction}_{opt.patch_size}"
        opt.profile_dir = f"{opt.profile_dir}/{opt.data_name}_{opt.arch}"
        opt.profile_dir += f"_{opt.model_construction}_{opt.patch_size}"
    else:
        opt.log_dir = f"{opt.log_dir}/{opt.save_name}"
        opt.profile_dir = f"{opt.profile_dir}/{opt.save_name}"

    opt.log_dir += f"_run1"
    opt.profile_dir += f"_run1"

    if os.path.exists(opt.log_dir):
        opt.log_dir = opt.log_dir[:-5]
        opt.profile_dir = opt.profile_dir[:-5]
        for i in range(1, 99999):
            if not os.path.exists(opt.log_dir+'_run'+str(i)):
                opt.log_dir = opt.log_dir+'_run'+str(i)
                opt.profile_dir = opt.profile_dir+'_'+str(i)
                break

    dist.broadcast_object_list([opt.log_dir, opt.profile_dir], 0)

    os.makedirs(opt.log_dir, exist_ok=True)

    fname = opt.data_name + '.csv'
    fname_val = opt.data_name + '_validation' + '.csv'

    result_dir = os.path.join(opt.log_dir, 'results/')
    res_train_dir = os.path.join(opt.log_dir, 'results/train/')
    res_val_dir = os.path.join(opt.log_dir, 'results/validation/')
    model_dir  = os.path.join(opt.log_dir, 'models')
    utils.mkdir(result_dir)
    utils.mkdir(res_train_dir)
    utils.mkdir(res_val_dir)
    utils.mkdir(model_dir)

    train_dir = os.path.join(opt.data_dir,'training')
    val_dir = os.path.join(opt.data_dir,'validation')
    print("====="*10)
    print(train_dir)
    print(val_dir)
    print("====="*10)

    return train_dir, val_dir, res_train_dir, res_val_dir, model_dir


def set_optimizer(opt, optim, model_restoration):
    if opt.optimizer.lower() == 'adam':
        optimizer = optim.Adam(model_restoration.parameters(), lr=opt.lr_initial, betas=(0.9, 0.999), eps=1e-6, weight_decay=opt.weight_decay)
    elif opt.optimizer.lower() == 'adamw':
        optimizer = optim.AdamW(model_restoration.parameters(), lr=opt.lr_initial, betas=(0.9, 0.999), eps=1e-6, weight_decay=opt.weight_decay)
    else:
        raise Exception("Error optimizer...")
    return optimizer


def warmup_scheduler(opt, optim, optimizer):
    if opt.warmup:
        print("Using warmup and cosine strategy!")
        warmup_epochs = opt.warmup_epochs
        scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.nepoch-warmup_epochs, eta_min=1e-6)
        scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
    else:
        print("Using cosine strategy!")
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.nepoch, eta_min=1e-6)

    return scheduler

def log_reviser(opt, epoch):
    
    data = None    
    with open(opt.log_dir, 'r') as f:
        data = f.readlines()

    try:
        os.remove(opt.log_dir)
    except:
        pass

    with open(opt.log_dir, 'w') as f2:
        for lineIdx in range(len(data)):
            if lineIdx >= (epoch-1):
                break
    
            f2.write(data[lineIdx])


def resume_ckpt(opt, model_restoration, scheduler):
    
    path_chk_rest = opt.pretrain_weights 
    print("Resume from "+path_chk_rest)
    utils.load_checkpoint(model_restoration,path_chk_rest) 
    best_epoch, best_psnr = utils.load_start_param(path_chk_rest) 
    start_epoch = best_epoch + 1

    log_reviser(opt, start_epoch)
    for i in range(1, start_epoch):
        scheduler.step()
    new_lr = scheduler.get_lr()[0]
    print('------------------------------------------------------------------------------')
    print("==> Resuming Training with learning rate:", new_lr)
    print('------------------------------------------------------------------------------')

    return start_epoch, new_lr, best_psnr

