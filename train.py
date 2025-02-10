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


def sync_and_empty_cache():
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

def print_memory_stats(name):
    torch.cuda.synchronize()
    memory_stats = torch.cuda.memory_stats()
    print(f"{name} allocated: {memory_stats['allocated_bytes.all.current'] / (1024 ** 2):.5f} MB, reserverd: {memory_stats['reserved_bytes.all.current'] / (1024 ** 2):.5f} MB")


def count_inf_nan(model):
    parameters = [p for p in model.parameters() if p.grad is not None]
    cnt_inf, cnt_nan = 0, 0
    for t in parameters:
        cnt_inf += torch.sum(torch.isinf(t)).item()
        cnt_nan += torch.sum(torch.isnan(t)).item()
    return cnt_inf, cnt_nan

def cleanup_grad(model):
    parameters = [p for p in model.parameters() if p.grad is not None]
    num_nans, num_infs = 0, 0
    for p in parameters:
        p_grad_= p.grad
        nan_idxs = torch.isnan(p_grad_)
        inf_idxs = torch.isinf(p_grad_)
        p_grad_[nan_idxs] = 0
        p_grad_[inf_idxs] = 0
        num_nans += torch.sum(nan_idxs)
        num_infs += torch.sum(inf_idxs)

    return num_nans, num_infs


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


def train_epoch(opt, epoch, model_restoration, optimizer, train_loader,
                scaler, losses, lr_scheduler, summarywriter, model_dir):
    model_restoration.train()
    train_loader.sampler.set_epoch(epoch)
    epoch_start_time = time.time()

    epoch_loss, epoch_loss_content, epoch_loss_ssim, epoch_loss_fft = 0, 0, 0, 0
    epoch_loss_fft_abs, epoch_loss_fft_angle, epoch_loss_fft_real, epoch_loss_fft_imag = 0, 0, 0, 0
    psnr_train_rgb, rmse_train_rgb, ssim_train_rgb = [], [], []

    for i, data in enumerate(tqdm(train_loader, disable=world_rank != 0), 0):
        optimizer.zero_grad()
        if opt.profile and i > 10: break
        target, input_ = data[0].cuda(), data[1].cuda()
        num_img = target.size()[0]

        if opt.precision == "fp32":
            restored = model_restoration(input_)
        elif opt.precision == "mixed":
            with torch.cuda.amp.autocast():
                restored = model_restoration(input_)

        targets = interpolate_down(target, opt.num_levels)
        
        loss, loss_content, loss_ssim, loss_fft, loss_fft_abs, loss_fft_angle, loss_fft_real, loss_fft_imag = ensemble_loss(opt, losses, restored, targets)

        if opt.precision == "fp32":
            loss.backward()
            # num_nans, num_infs = cleanup_grad(model_restoration)
            optimizer.step()
        elif opt.precision == "mixed":
            scaler.scale(loss).backward()

            # scaler.unscale_(optimizer)
            # torch.nn.utils.clip_grad_norm_(model_restoration.parameters(), 2.0)
            # num_nans, num_infs = cleanup_grad(model_restoration)

            num_infs, num_nans = count_inf_nan(model_restoration)
            if num_nans > 0 or num_infs > 0:
                trainable_params = sum(
                    p.numel() for p in model_restoration.parameters() if p.requires_grad
                )
                print("{}-{} NaNs: {}/{}, Infs: {}/{}".format(epoch, i, num_nans, trainable_params, num_infs, trainable_params))
            scaler.step(optimizer)
            scaler.update()

        if opt.mse_scale > 0.0:
            epoch_loss_content += loss_content.item()
        if opt.ssim_scale > 0.0:
            epoch_loss_ssim += loss_ssim.item()
        if opt.fft_scale > 0.0:
            epoch_loss_fft += loss_fft.item()
        if opt.fft_scale_abs > 0.0:
            epoch_loss_fft_abs += loss_fft_abs.item()
        if opt.fft_scale_angle > 0.0:
            epoch_loss_fft_angle += loss_fft_angle.item()
        if opt.fft_scale_real > 0.0:
            epoch_loss_fft_real += loss_fft_real.item()
        if opt.fft_scale_imag > 0.0:
            epoch_loss_fft_imag += loss_fft_imag.item()

        epoch_loss += loss.item()

        if isinstance(restored, list):
            restored = restored[-1]

        restored = torch.clamp(restored, 0, 1)

        psnr, rmse = utils.batch_PSNR(restored, target)
        ssim = utils.calc_metric(restored, target, size_average=False).tolist()

        psnr_train_rgb = psnr_train_rgb + psnr
        rmse_train_rgb = rmse_train_rgb + rmse
        ssim_train_rgb = ssim_train_rgb + ssim

    # All-reduce metrics across all processes
    psnr_train_rgb = mp_all_reduce_mean(psnr_train_rgb)
    ssim_train_rgb = mp_all_reduce_mean(ssim_train_rgb)
    rmse_train_rgb = mp_all_reduce_mean(rmse_train_rgb)

    time_per_epoch = time.time() - epoch_start_time
    print(
        "[Training] Ep %d | PSNR-tr=%.4f (%.4f) | SSIM-tr=%.4f | Loss=%.4f | Time=%.2fs  ----  ( LR: %.5e, Loss Scale: %.5e )"
        % (epoch, psnr_train_rgb, rmse_train_rgb, ssim_train_rgb, epoch_loss, time_per_epoch, lr_scheduler.get_last_lr()[0], scaler.get_scale())
    )

    if iam_first:
        summarywriter.add_scalar("train/Loss", epoch_loss, epoch)
        summarywriter.add_scalar("train/PSNR", psnr_train_rgb, epoch)
        summarywriter.add_scalar("train/SSIM", ssim_train_rgb, epoch)
        summarywriter.add_scalar("train/RMSE", rmse_train_rgb, epoch)
        summarywriter.add_scalar("train/time", time_per_epoch, epoch)
        summarywriter.add_scalar("train/lr", lr_scheduler.get_last_lr()[0], epoch)
        torch.save(
            {
                "epoch": epoch,
                "best_psnr": psnr_train_rgb,
                "state_dict": model_restoration.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
            },
            os.path.join(model_dir, "model_latest.pth"),
        )

        if epoch % opt.ckpt == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "best_psnr": psnr_train_rgb,
                    "state_dict": model_restoration.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                },
                os.path.join(model_dir, "model_epoch_{}.pth".format(epoch)),
            )



def val_epoch(opt, epoch, model_restoration, optimizer, val_loader, scaler,
              summarywriter, model_dir, img_save_dir, best_psnr, best_epoch):
    #### Evaluation ####
    val_loader.sampler.set_epoch(epoch)
    model_restoration.eval()
    psnr_val_rgb, ssim_val_rgb = [], []

    for i, data_val in enumerate(tqdm(val_loader, disable=world_rank != 0), 0):
        torch.cuda.empty_cache()
        target, input_, target_fnames, restored_fnames \
            = data_val[0], data_val[1].cuda(), data_val[2], data_val[3]
        num_img = target.size()[0]

        if opt.profile and i > 1:
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

        target = target.cuda()
        psnr, rmse = utils.batch_PSNR(restored, target)
        ssim = utils.calc_metric(restored, target, size_average=False).tolist()

        psnr_val_rgb = psnr_val_rgb + psnr
        ssim_val_rgb = ssim_val_rgb + ssim

        if epoch % opt.save_epoch == 0 and iam_first:
            save_output(opt, restored, epoch, img_save_dir, restored_fnames)


    # All-reduce metrics across all processes
    psnr_val_rgb = mp_all_reduce_mean(psnr_val_rgb)
    ssim_val_rgb = mp_all_reduce_mean(ssim_val_rgb)

    if iam_first:
        summarywriter.add_scalar("val/PSNR", psnr_val_rgb, epoch)
        summarywriter.add_scalar("val/SSIM", ssim_val_rgb, epoch)

        if psnr_val_rgb > best_psnr:
            best_psnr = psnr_val_rgb
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "best_psnr": best_psnr,
                    "state_dict": model_restoration.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                },
                os.path.join(model_dir, "model_best.pth"),
            )

    print("[Validation] Node %s | Ep %d | PSNR-val=%.4f | SSIM-val=%.4f  ----  [ best_Ep %d | Best_PSNR: %.4f ]"
        % (os.uname()[1], epoch, psnr_val_rgb, ssim_val_rgb, best_epoch, best_psnr))

    return best_psnr, best_epoch

def train(opt):
    train_dir, val_dir, res_train_dir, res_val_dir, model_dir = utils.set_env.set_dirs(opt)
    print(opt)

    writer = SummaryWriter(log_dir=opt.log_dir) if iam_first else None
    export_options_to_summarywriter(writer, opt)

    train_loader, val_loader, len_trainset, len_valset, train_dataset = load_dataset(opt, train_dir, val_dir)

    opt.C, opt.H, opt.W = train_dataset.return_size()
    
    """ SFIM Model & Optimizer Definition """
    # assert opt.arch == 'SFIM'
    if opt.arch == 'SFIM':
        model_restoration = SFIM(opt)
    model_restoration = model_restoration.cuda()
    model_restoration = DDP(model_restoration, device_ids=[local_rank], static_graph=True)
    optimizer = utils.set_env.set_optimizer(opt, optim, model_restoration)
    scheduler = utils.set_env.warmup_scheduler(opt, optim, optimizer)
    start_epoch = 1
    best_psnr, best_epoch = 0, 0

    """ Resume """
    if opt.resume_from is not None:
        print("Loading model from {}".format(opt.resume_from))
        checkpoint = torch.load(opt.resume_from, map_location="cpu")
        model_restoration.load_state_dict(checkpoint["state_dict"], strict=False)
        # optimizer.load_state_dict(checkpoint["optimizer"])
        best_psnr = checkpoint["best_psnr"]
        best_epoch = checkpoint["epoch"]
        print(f"Saved epoch: {best_epoch}, best PSNR: {best_psnr:.4f}")

        del checkpoint

    start_epoch = opt.start_epoch
    end_epoch = start_epoch + opt.nepoch

    """ Loss Calculation """
    loss_Charbonier = CharbonnierLoss().cuda()
    loss_SSIM = SSIM_Loss().cuda()
    loss_FFT = FFT_loss().cuda()
    loss_FFT_abs = FFT_abs_L1().cuda()
    loss_FFT_angle = FFT_angle_L1().cuda()
    loss_FFT_real = FFTLoss_real().cuda()
    loss_FFT_imag = FFTLoss_imag().cuda()
    losses = [loss_Charbonier, loss_SSIM, loss_FFT, loss_FFT_abs, loss_FFT_angle, loss_FFT_real, loss_FFT_imag]

    if opt.profile:
        prof = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=2, active=2, repeat=1),
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            on_trace_ready=torch.profiler.tensorboard_trace_handler(opt.profile_dir),
        )
        prof.start()

    print("\n===> Start Epoch {}, End Epoch {}".format(start_epoch, end_epoch))
    scaler = torch.cuda.amp.GradScaler()

    """ Evaluate before training if resume. For sanity check """
    if start_epoch > 1:
        best_psnr, best_epoch = val_epoch(opt, start_epoch - 1, model_restoration, optimizer, val_loader, scaler, writer, model_dir, res_val_dir, best_psnr, best_epoch)
    sync_and_empty_cache()
    for epoch in range(start_epoch, end_epoch):
        sync_and_empty_cache()
        print_memory_stats(f"Before train {epoch}")
        train_epoch(opt, epoch, model_restoration, optimizer, train_loader, scaler, losses, scheduler, writer, model_dir)
        print_memory_stats(f"After train {epoch}")
        if epoch % opt.val_epoch == 0:
            sync_and_empty_cache()
            print_memory_stats(f"Before val {epoch}")
            best_psnr, best_epoch = val_epoch(opt, epoch, model_restoration, optimizer, val_loader, scaler, writer, model_dir, res_val_dir, best_psnr, best_epoch)
            print_memory_stats(f"After val {epoch}")
        scheduler.step()


    if opt.profile:
        prof.stop()


def load_dataset(opt, train_dir, val_dir):
    print("\n===> Loading datasets for DDP Training")
    train_dataset = dataset.DataLoaderTrain(
        opt, train_dir, opt.data_format, opt.patch_size
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=False,
        drop_last=True, # Required to be true for static batch sizes. It is OK to skip some samples in training.
        sampler=DistributedSampler(train_dataset, shuffle=True, drop_last=False),
    )

    val_dataset = dataset.DataLoaderVal(opt, val_dir, opt.data_format)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=False,
        drop_last=False, # Do not skip any sample in validation.
        sampler=DistributedSampler(val_dataset, shuffle=True, drop_last=False),
    )

    len_trainset = train_dataset.__len__()
    len_valset = val_dataset.__len__()
    print(
        "Sizeof training set: ", len_trainset, ", sizeof validation set: ", len_valset
    )

    return train_loader, val_loader, len_trainset, len_valset, train_dataset


def interpolate_down(x, num_levels):
    x_2 = F.interpolate(x, scale_factor=0.5)  # 1, 4, 128, 128
    x_4 = F.interpolate(x_2, scale_factor=0.5)  # 1, 4, 64, 64
    x_8 = F.interpolate(x_4, scale_factor=0.5)  # 1, 4, 32, 32
    if num_levels == 4:
        return [x_8, x_4, x_2, x]
    if num_levels == 3:
        return [x_4, x_2, x]
    if num_levels == 2:
        return [x_2, x]
    if num_levels == 1:
        return [x]


def calculate_loss(opt, losses, lq, gt):
    loss, loss_content, loss_ssim, loss_fft, loss_fft_abs, loss_fft_angle, loss_fft_real, loss_fft_imag = (
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    )

    if opt.mse_scale > 0.0:
        loss_content = losses[0](lq, gt)
        loss = loss_content * opt.mse_scale
    if opt.ssim_scale > 0.0:
        loss_ssim = losses[1](lq, gt)
        loss += loss_ssim * opt.ssim_scale
    if opt.fft_scale > 0.0:
        loss_fft = losses[2](lq, gt)
        loss += loss_fft * opt.fft_scale
    if opt.fft_scale_abs > 0.0:
        loss_fft_abs = losses[3](lq, gt)
        loss += loss_fft_abs * opt.fft_scale_abs
    if opt.fft_scale_angle > 0.0:
        loss_fft_angle = losses[4](lq, gt)
        loss += loss_fft_angle * opt.fft_scale_angle
    if opt.fft_scale_real > 0.0:
        loss_fft_real = losses[5](lq, gt)
        loss += loss_fft_real * opt.fft_scale_real
    if opt.fft_scale_imag > 0.0:
        loss_fft_imag = losses[6](lq, gt)
        loss += loss_fft_imag * opt.fft_scale_imag
    return (
        loss,
        loss_content,
        loss_ssim,
        loss_fft,
        loss_fft_abs,
        loss_fft_angle,
        loss_fft_real,
        loss_fft_imag,
    )


def ensemble_loss(opt, losses, lqs, gts):
    total_loss = 0
    total_loss_content = 0
    total_loss_ssim = 0
    total_loss_fft = 0
    total_loss_fft_abs = 0
    total_loss_fft_angle = 0
    total_loss_fft_real = 0
    total_loss_fft_imag = 0

    for lq, gt in zip(lqs, gts):
        (
            loss,
            loss_content,
            loss_ssim,
            loss_fft,
            loss_fft_abs,
            loss_fft_angle,
            loss_fft_real,
            loss_fft_imag,
        ) = calculate_loss(opt, losses, lq, gt)
        total_loss += loss
        total_loss_content += loss_content
        total_loss_ssim += loss_ssim
        total_loss_fft += loss_fft
        total_loss_fft_abs += loss_fft_abs
        total_loss_fft_angle += loss_fft_angle
        total_loss_fft_real += loss_fft_real
        total_loss_fft_imag += loss_fft_imag

    return (
        total_loss,
        total_loss_content,
        total_loss_ssim,
        total_loss_fft,
        total_loss_fft_abs,
        total_loss_fft_angle,
        total_loss_fft_real,
        total_loss_fft_imag,
    )


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

    train(opt)


if __name__ == "__main__":
    main()
