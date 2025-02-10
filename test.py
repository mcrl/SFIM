import os
import argparse
import options
import time
import datetime

import utils
from timm.utils import NativeScaler


import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from utils.image_utils import save_output, calc_metric
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from utils.model_utils import print_memory_stats

from networks.sfim import SFIM
from options import export_options_to_summarywriter

from tqdm import tqdm

from losses import *
import dataset
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure

world_rank = int(os.environ.get("OMPI_COMM_WORLD_RANK"))
local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK"))
local_size = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_SIZE"))
world_size = int(
    os.environ.get("OMPI_COMM_WORLD_SIZE", os.environ.get("WORLD_SIZE", -1))
)
iam_first = world_rank == 0
iam_last = world_rank == world_size - 1

def mp_all_reduce_mean(vals):
    vals_sum_tensor = torch.Tensor([sum(vals)]).cuda()
    vals_len_tensor = torch.Tensor([len(vals)]).cuda()
    dist.all_reduce(vals_sum_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(vals_len_tensor, op=dist.ReduceOp.SUM)
    return vals_sum_tensor.item() / vals_len_tensor.item()

def test(opt):
    print(opt)

    test_loader, len_testset, test_dataset = load_dataset(opt, opt.data_dir + '/test')

    opt.C, opt.H, opt.W = test_dataset.return_size()

    """ SFIM Model & Optimizer Definition """
    assert opt.arch == 'SFIM'
    model_restoration = SFIM(opt)
    model_restoration = model_restoration.cuda()
    model_restoration = DDP(model_restoration, device_ids=[local_rank], static_graph=True)

    """ Resume """
    if opt.resume_from is not None:
        print("Loading model from {}".format(opt.resume_from))
        checkpoint = torch.load(opt.resume_from, map_location="cpu")
        model_restoration.load_state_dict(checkpoint["state_dict"], strict=False)
        print("Model loaded from {}".format(opt.resume_from))


    test_loader.sampler.set_epoch(1)
    model_restoration.eval()
    psnr_val_rgb, ssim_val_rgb = [], []

    for ii, data_val in enumerate(tqdm(test_loader, disable=world_rank != 0), 0):
        target, input_, target_fnames, restored_fnames \
            = data_val[0].cuda(), data_val[1].cuda(), data_val[2], data_val[3]
        num_img = target.size()[0]

        if opt.profile and ii > 1:
            break

        with torch.no_grad():
            if opt.precision == "fp32":
                restored = model_restoration(input_)
            elif opt.precision == "mixed":
                with torch.cuda.amp.autocast():
                    restored = model_restoration(input_)

        if isinstance(restored, list):
            restored = restored[-1]

        restored = torch.clamp(restored, 0, 1)

        restored_uint8 = torch.clamp(restored * 255, 0, 255).to(torch.uint8).to(torch.float32)
        target_uint8 = torch.clamp(target * 255, 0, 255).to(torch.uint8).to(torch.float32)

        psnr = PeakSignalNoiseRatio(data_range=255.0).cuda()(restored_uint8, target_uint8)
        ssim = StructuralSimilarityIndexMeasure(data_range=255.0).cuda()(restored_uint8, target_uint8)

        psnr_val_rgb = psnr_val_rgb + [psnr.item()]
        ssim_val_rgb = ssim_val_rgb + [ssim.item()]

        restored = torch.clamp(restored, 0, 1) # Need clamp before saving images. 
        save_output(opt, restored, -1, opt.restored_img_dir, restored_fnames, None, psnr.item(), ssim.item())

    # All-reduce metrics across all processes
    psnr_val_rgb = mp_all_reduce_mean(psnr_val_rgb)
    ssim_val_rgb = mp_all_reduce_mean(ssim_val_rgb)

    print(f"Test Result:")
    print(f"  - PSNR: {psnr_val_rgb:.4f}")
    print(f"  - SSIM: {ssim_val_rgb:.4f}")


def load_dataset(opt, test_dir):
    print("\n===> Loading datasets for Testing")
    test_dataset = dataset.DataLoaderTest(opt, test_dir, opt.data_format)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=False,
        drop_last=False,
        sampler=DistributedSampler(test_dataset, shuffle=False),
    )

    len_testset = test_dataset.__len__()
    return test_loader, len_testset, test_dataset


def interpolate_down(x):
    x_2 = F.interpolate(x, scale_factor=0.5)  # 1, 4, 128, 128
    x_4 = F.interpolate(x_2, scale_factor=0.5)  # 1, 4, 64, 64
    x_8 = F.interpolate(x_4, scale_factor=0.5)  # 1, 4, 32, 32
    return [x_8, x_4, x_2, x]



def setup_distributed():
    dist.init_process_group("nccl", rank=world_rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    print(
        "Node %s | World size: %d | Local rank: %d | World rank: %d"
        % (os.uname()[1], world_size, local_rank, world_rank)
    )
    if world_rank != 0:
        import sys
        sys.stdout = open(os.devnull, "w")


def main():
    torch.backends.cudnn.benchmark = False
    setup_distributed()
    opt = (
        options.Options()
        .init(argparse.ArgumentParser(description="UDC image restoration"))
        .parse_args()
    )
    utils.set_env.set_seeds(1234)
    opt = utils.set_env.automatic_opt_setting(opt)

    test(opt)


if __name__ == "__main__":
    main()
