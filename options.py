import os
import torch
import argparse


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Options():
    def __init__(self):
        pass

    def init(self, parser):
        # basic settings
        parser.add_argument('--patch_size', type=int, default=256, help='patch size of training sample') # ONLY used for batch_mode = 0
        parser.add_argument('--patch_size_val', type=int, default=256, help='patch size of validation sample')
        parser.add_argument('--H', type=int, default=512, help='height of input image')
        parser.add_argument('--W', type=int, default=512, help='width of input image')
        parser.add_argument('--C', type=int, default=4, help='channel of input image')

        # loss function
        parser.add_argument('--mse_scale', type=float, default=1.0, help='scale value for loss_mse')
        parser.add_argument('--ssim_scale', type=float, default=0.0, help='scale value for loss_ssim')
        parser.add_argument('--fft_scale', type=float, default=0.0, help='scale value for loss_fft')
        parser.add_argument('--fft_scale_abs', type=float, default=1.0, help='scale value for loss_fft')
        parser.add_argument('--fft_scale_angle', type=float, default=1.0, help='scale value for loss_fft')
        parser.add_argument('--fft_scale_real', type=float, default=0.0, help='scale value for loss_fft')
        parser.add_argument('--fft_scale_imag', type=float, default=0.0, help='scale value for loss_fft')

        parser.add_argument('--precision', type=str, default="fp32", help='precision: mixed or fp32')
        parser.add_argument('--save_epoch', type=int, default=10, help='save epoch for restored images')
        parser.add_argument('--perm_args', type=str, default='2,0,1', help='permute pattern for UDC_SIT images')
        parser.add_argument('--max_pxl', type=float, default=1023.0, help='max pixel value of input images')
        parser.add_argument('--source_dir', type=str, default='./background.dng', help='source_dir for save to png of 4 channel images')
        parser.add_argument('--tonemap', type=str2bool, default=False, help='Tone Mapping for HDR images')
        parser.add_argument('--val_epoch', type=int, default=10, help='validation epoch')

        # Network backbone: FFTformer
        parser.add_argument('--num_FFTblock', type=int, default=6, help='num_FFTblock')
        parser.add_argument('--ffn_expansion_factor', type=int, default=3, help='ffn_expansion_factor')

        # Network general
        parser.add_argument('--apply_CASA', type=str2bool, default=True, help='apply CA for all levels')

        # global settings
        parser.add_argument('--batch_size', type=int, default=32, help='batch size')
        parser.add_argument('--nepoch', type=int, default=250, help='training epochs')
        parser.add_argument('--pretrain_weights',type=str, default='./logs/udc/Tformer/models/model_best.pth', help='path of pretrained_weights')
        parser.add_argument('--optimizer', type=str, default ='adamw', help='optimizer for training')
        parser.add_argument('--lr_initial', type=float, default=0.0002, help='initial learning rate')
        parser.add_argument('--step_lr', type=int, default=50, help='weight decay')
        parser.add_argument('--weight_decay', type=float, default=0.02, help='weight decay')
        parser.add_argument('--arch', type=str, default ='sfim',  help='archtechture')
        parser.add_argument('--num_ch', type=int, default=4, help='num_ch') # Feng: 3, UDC-SIT: 4 (or 3 for jpg)

        # args for saving
        parser.add_argument('--save_images', action='store_true',default=False)
        parser.add_argument('--ckpt', type=int, default=50, help='checkpoint')
        parser.add_argument('--save_name', type=str, default='', help='')

        # args for dataset
        parser.add_argument('--data_dir', type=str, default ='/data/s0/udc/dataset/UDC_SIT/npy/', help='directory of dataset')
        parser.add_argument('--data_name', type=str, default ='UDC_SIT') # Feng, UDC_SIT, ...
        parser.add_argument('--data_format', type=int, default=0, help='0 : npy, 1 : png, 2 : jpg, 3 : pkl')

        # args for Thunder Transformer (it was in Uformer)
        parser.add_argument('--norm_layer', type=str, default ='nn.LayerNorm', help='normalize layer in transformer')
        parser.add_argument('--embed_dim', type=int, default=48, help='dim of emdeding features')
        
        # args for training
        parser.add_argument('--resume', action='store_true',default=False)
        parser.add_argument('--warmup', action='store_true', default=False, help='warmup') 
        parser.add_argument('--warmup_epochs', type=int,default=3, help='epochs for warmup')
        parser.add_argument('--start_epoch', type=int, default=1, help='initial epoch number for progressive training')

        parser.add_argument('--profile', type=str2bool, default=False, help='run pytorch profiling')
        parser.add_argument('--log_dir', type=str, default ='/shared/s1/lab08/udc_logs', help='directory to store tensorboard logs')
        parser.add_argument('--profile_dir', type=str, default ='/shared/s1/lab08/udc_profiles', help='directory to store tensorboard logs')

        # args for testing
        parser.add_argument('--restored_img_dir', type=str, default='./restored_images/')
        parser.add_argument('--resume_from', type=str, default=None)

        parser.add_argument('--num_levels', type=int, default=4)


        return parser

def export_options_to_summarywriter(writer, opt):
    if writer is not None:
        writer.add_hparams({
            'patch_size': opt.patch_size,
            'patch_size_val': opt.patch_size_val,
            'precision': opt.precision,
            'save_epoch': opt.save_epoch,
            'val_epoch': opt.val_epoch,
            'num_FFTblock': opt.num_FFTblock,
            'ffn_expansion_factor': opt.ffn_expansion_factor,
            'apply_CASA': opt.apply_CASA,
            'batch_size': opt.batch_size,
            'optimizer': opt.optimizer,
            'lr_initial': opt.lr_initial,
            'arch': opt.arch,
            'norm_layer': opt.norm_layer,
            'embed_dim': opt.embed_dim,
            'log_dir': opt.log_dir,
            'profile_dir': opt.profile_dir,
        },
        {}
        )
