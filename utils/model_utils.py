import torch
import torch.nn as nn
import os
from collections import OrderedDict


def save_checkpoint(model_dir, state, session):
    epoch = state['epoch']
    model_out_path = os.path.join(model_dir,"model_epoch_{}_{}.pth".format(epoch,session))
    torch.save(state, model_out_path)

def load_checkpoint(model, weights):
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if 'module.' in k else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)


def load_checkpoint_multigpu(model, weights):
    checkpoint = torch.load(weights)
    state_dict = checkpoint["state_dict"]
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] 
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

def load_start_param(weights):
    checkpoint = torch.load(weights)
    epoch = checkpoint["epoch"]
    try:
        best_psnr = checkpoint["best_psnr"]
    except:
        print("Load from old params")
        best_psnr = 0
    return epoch, best_psnr

def load_optim(optimizer, weights):
    checkpoint = torch.load(weights)
    optimizer.load_state_dict(checkpoint['optimizer'])
    for p in optimizer.param_groups: lr = p['lr']
    return lr

def print_memory_stats(name):
    torch.cuda.synchronize()
    memory_stats = torch.cuda.memory_stats()
    print(f"{name} allocated: {memory_stats['allocated_bytes.all.current'] / 1024 ** 3:.1f} GB, reserverd: {memory_stats['reserved_bytes.all.current'] / 1024 ** 3:.1f} GB")
