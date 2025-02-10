import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .basicConv import BasicConv
from .cbam import CBAM
import numbers
from einops import rearrange
import torch.utils.checkpoint as checkpoint

def count_inf_nan(t):
    cnt_inf, cnt_nan = 0, 0
    cnt_inf = torch.sum(torch.isinf(t)).item()
    cnt_nan = torch.sum(torch.isnan(t)).item()
    return cnt_inf, cnt_nan

def print_memory_stats(name):
    return
    torch.cuda.synchronize()
    memory_stats = torch.cuda.memory_stats()
    print(f"{name} allocated: {memory_stats['allocated_bytes.all.current'] / 1024 ** 3:.1f} GB, reserverd: {memory_stats['reserved_bytes.all.current'] / 1024 ** 3:.1f} GB")



def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G = growRate
        self.conv = nn.Sequential(
            *[
                nn.Conv2d(Cin, G, kSize, padding=(kSize - 1) // 2, stride=1),
                nn.GELU(),
            ]
        )

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)


class RDBlock(nn.Module):
    def __init__(self, in_channel, out_channel, nConvLayers=3):
        super(RDBlock, self).__init__()
        G0 = in_channel
        G = in_channel
        C = nConvLayers

        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c * G, G))
        self.convs = nn.Sequential(*convs)
        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C * G, out_channel, 1, padding=0, stride=1)

    def forward(self, x):
        return self.LFF(self.convs(x)) + x


class EBlock(nn.Module):
    def __init__(self, out_channel, num_res=8):
        super(EBlock, self).__init__()
        layers = [RDBlock(out_channel, out_channel) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DBlock(nn.Module):
    def __init__(self, channel, num_res=8):
        super(DBlock, self).__init__()
        layers = [RDBlock(channel, channel) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

###############################################################################

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        b, c, h, w = x.shape

        return to_4d(self.body(to_3d(x)), h, w)


class DFFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias, gt_size=(128,128)):

        super(DFFN, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.patch_size = 8

        self.dim = dim
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.fft = nn.Parameter(torch.ones((hidden_features * 2, 1, 1, self.patch_size, self.patch_size // 2 + 1))) 
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        if x.shape[2] % 8 != 0: 
            PAD = True
            pad_x, pad_y = int((x.shape[2] % 8) / 2), int((x.shape[3] % 8) / 2)
            x = F.pad(x, (pad_y, pad_y, pad_x, pad_x), 'reflect')
        else:
            PAD = False
        x_patch = rearrange(x, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        x_patch_fft = torch.fft.rfft2(x_patch)
        x_patch_fft = x_patch_fft * self.fft
        x_patch = torch.fft.irfft2(x_patch_fft, s=(self.patch_size, self.patch_size)) # torch.abs(x_patch_fft)
        x = rearrange(x_patch, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                      patch2=self.patch_size)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)

        x = F.gelu(x1) * x2
        x = self.project_out(x)

        if PAD: 
            x = x[:, :, pad_x : -pad_x, pad_y : -pad_y]

        return x


class FSAS(nn.Module):
    def __init__(self, dim, bias):
        super(FSAS, self).__init__()

        self.to_hidden = nn.Conv2d(dim, dim * 6, kernel_size=1, bias=bias)
        self.to_hidden_dw = nn.Conv2d(dim * 6, dim * 6, kernel_size=3, stride=1, padding=1, groups=dim * 6, bias=bias)

        self.project_out = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)

        self.norm = LayerNorm(dim * 2, LayerNorm_type='WithBias')

        self.patch_size = 8

    def forward(self, x):
        if x.shape[2] % 8 != 0: 
            PAD = True
            pad_x, pad_y = int((x.shape[2] % 8) / 2), int((x.shape[3] % 8) / 2)
            x = F.pad(x, (pad_y, pad_y, pad_x, pad_x), 'reflect')
        else:
            PAD = False

        hidden = self.to_hidden(x)

        q, k, v = self.to_hidden_dw(hidden).chunk(3, dim=1)

        q_patch = rearrange(q, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        k_patch = rearrange(k, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        
        q_fft = torch.fft.rfft2(q_patch)
        k_fft = torch.fft.rfft2(k_patch)

        out = q_fft * k_fft
        out = torch.fft.irfft2(out, s=(self.patch_size, self.patch_size)) 
        out = rearrange(out, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                        patch2=self.patch_size)

        out = self.norm(out)

        output = v * out

        if PAD: 
            output = output[:, :, pad_x : -pad_x, pad_y : -pad_y]

        output = self.project_out(output)

        return output


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias', att=False, gt_size=(128,128)):
        super(TransformerBlock, self).__init__()
        self.att = att
        if self.att:
            self.norm1 = LayerNorm(dim, LayerNorm_type)
            self.attn = FSAS(dim, bias)

        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = DFFN(dim, ffn_expansion_factor, bias, gt_size=gt_size)

    def forward(self, x):
        if self.att:
            x = x + self.attn(self.norm1(x))

        x = x + self.ffn(self.norm2(x))

        return x


##########################################################################
class MIB1(nn.Module):
    def __init__(self, in_channel, out_channel, ffn_expansion_factor=2, bias=False):
        super(MIB1, self).__init__()
        hidden_features = int(in_channel * ffn_expansion_factor)
        self.project_in = nn.Conv2d(
            in_channel, hidden_features * 2, kernel_size=1, bias=bias
        )
        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
            bias=bias,
        )

        self.project_out = nn.Conv2d(
            hidden_features * 2, out_channel, kernel_size=1, bias=bias
        )

    def forward(self, x1, x2, x3, x4):
        x = [tensor for tensor in [x1, x2, x3, x4] if tensor is not None]
        x = torch.cat(x, dim=1)
        del x1, x2, x3, x4
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x1 = x1.contiguous()
        x2 = x2.contiguous()
        del x

        x1_ = F.sigmoid(x1) * x2
        x2_ = F.sigmoid(x2) * x1
        del x1, x2

        x = torch.cat([x1_, x2_], dim=1)
        del x1_, x2_
        x = self.project_out(x)

        return x  # self.conv(x)



class MIB(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(MIB, self).__init__()
        self.conv = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False),
        )

    def forward(self, x1, x2, x3, x4):
        # x = torch.cat([x1, x2, x3, x4], dim=1)
        x = [tensor for tensor in [x1, x2, x3, x4] if tensor is not None]
        x = torch.cat(x, dim=1)
        return self.conv(x)


class SCM_enc(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(SCM_enc, self).__init__()
        self.main = nn.Sequential(
            BasicConv(in_ch, out_ch//4, kernel_size=3, stride=1, relu=True),
            BasicConv(out_ch // 4, out_ch // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_ch // 2, out_ch // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_ch // 2, out_ch-in_ch, kernel_size=1, stride=1, relu=True)
        )

        self.conv = BasicConv(out_ch, out_ch, kernel_size=1, stride=1, relu=False)

    def forward(self, x):
        x = torch.cat([x, self.main(x)], dim=1)
        return self.conv(x)


class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.merge = BasicConv(channel, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):
        x = x1 * x2
        out = x1 + self.merge(x)
        return out

class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size=3, in_nc=3):
        super(SAM, self).__init__()
        self.conv1 = BasicConv(
            n_feat, n_feat, kernel_size=kernel_size, stride=1, relu=True
        )
        self.conv2 = BasicConv(
            n_feat, in_nc, kernel_size=kernel_size, stride=1, relu=False
        )
        self.conv3 = BasicConv(
            in_nc, n_feat, kernel_size=kernel_size, stride=1, relu=False
        )

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1 * x2
        x1 = x1 + x
        return x1, img



class SFIM(nn.Module):
    def __init__(self, opt):
        super(SFIM, self).__init__()

        self.opt = opt
        self.embed_dim = opt.embed_dim
        self.patch_size = opt.patch_size
        self.num_blocks = opt.num_FFTblock
        self.gt_size = (opt.patch_size, opt.patch_size)
        ffn_expansion_factor = opt.ffn_expansion_factor

        def div_tuple(in_tuple, div):
            return tuple(x / div for x in in_tuple)

        self.Encoder = nn.ModuleList([
            EBlock(self.embed_dim, num_res=8) if opt.num_levels >= 1 else nn.Identity(),
            nn.Sequential(*[TransformerBlock(dim=int(self.embed_dim * 2 ** 1), ffn_expansion_factor=ffn_expansion_factor,
                                gt_size=div_tuple(self.gt_size, 2)) for i in range(self.num_blocks)]) if opt.num_levels >= 2 else nn.Identity(),
            nn.Sequential(*[TransformerBlock(dim=int(self.embed_dim * 2 ** 2), ffn_expansion_factor=ffn_expansion_factor,
                                gt_size=div_tuple(self.gt_size, 2 ** 2)) for i in range(self.num_blocks)]) if opt.num_levels >= 3 else nn.Identity(),
            nn.Sequential(*[TransformerBlock(dim=int(self.embed_dim * 2 ** 2), ffn_expansion_factor=ffn_expansion_factor,
                                gt_size=div_tuple(self.gt_size, 2 ** 3)) for i in range(self.num_blocks)]) if opt.num_levels >= 4 else nn.Identity(),
        ])
        self.Decoder = nn.ModuleList([
            nn.Sequential(*[TransformerBlock(dim=int(self.embed_dim * 2 ** 2), ffn_expansion_factor=ffn_expansion_factor,
                                att=True, gt_size=div_tuple(self.gt_size, 2 ** 3)) for i in range(self.num_blocks)]) if opt.num_levels >= 4 else nn.Identity(),
            nn.Sequential(*[TransformerBlock(dim=int(self.embed_dim * 2 ** 2), ffn_expansion_factor=ffn_expansion_factor,
                                att=True, gt_size=div_tuple(self.gt_size, 2 ** 2)) for i in range(self.num_blocks)]) if opt.num_levels >= 3 else nn.Identity(),
            nn.Sequential(*[TransformerBlock(dim=int(self.embed_dim * 2 ** 1), ffn_expansion_factor=ffn_expansion_factor,
                                att=True, gt_size=div_tuple(self.gt_size, 2)) for i in range(self.num_blocks)]) if opt.num_levels >= 2 else nn.Identity(),
            DBlock(self.embed_dim, num_res=8) if opt.num_levels >= 1 else nn.Identity(),
        ])

        self.FEM = nn.ModuleList([
            BasicConv(opt.num_ch, self.embed_dim * 1, kernel_size=3, relu=True, stride=1) if opt.num_levels >= 1 else nn.Identity(),
            BasicConv(self.embed_dim, self.embed_dim * 2, kernel_size=3, relu=True, stride=2) if opt.num_levels >= 2 else nn.Identity(),
            BasicConv(self.embed_dim * 2, self.embed_dim * 4, kernel_size=3, relu=True, stride=2) if opt.num_levels >= 3 else nn.Identity(),
            BasicConv(self.embed_dim * 4, self.embed_dim * 4, kernel_size=3, relu=True, stride=2) if opt.num_levels >= 4 else nn.Identity(),
            BasicConv(self.embed_dim * 4, self.embed_dim * 4, kernel_size=4, relu=True, stride=2, transpose=True,) if opt.num_levels >= 4 else nn.Identity(),
            BasicConv(self.embed_dim * 4, self.embed_dim * 2, kernel_size=4, relu=True, stride=2, transpose=True) if opt.num_levels >= 3 else nn.Identity(),
            BasicConv(self.embed_dim * 2, self.embed_dim, kernel_size=4, relu=True, stride=2, transpose=True) if opt.num_levels >= 2 else nn.Identity(),
            BasicConv(self.embed_dim * 1, opt.num_ch, kernel_size=3, relu=False, stride=1) if opt.num_levels >= 1 else nn.Identity(),
        ])

        self.SCM2 = SCM_enc(opt.num_ch, self.embed_dim * 2) if opt.num_levels >= 2 else nn.Identity()
        self.SCM3 = SCM_enc(opt.num_ch, self.embed_dim * 4) if opt.num_levels >= 3 else nn.Identity()
        self.SCM4 = SCM_enc(opt.num_ch, self.embed_dim * 4) if opt.num_levels >= 4 else nn.Identity()

        self.FAM2 = FAM(self.embed_dim * 2) if opt.num_levels >= 2 else nn.Identity()
        self.FAM3 = FAM(self.embed_dim * 4) if opt.num_levels >= 3 else nn.Identity()
        self.FAM4 = FAM(self.embed_dim * 4) if opt.num_levels >= 4 else nn.Identity()

        if opt.apply_CASA:
            self.CASA1 = CBAM(self.embed_dim * 1) if opt.num_levels >= 1 else nn.Identity()
            self.CASA2 = CBAM(self.embed_dim * 2) if opt.num_levels >= 2 else nn.Identity()
            self.CASA3 = CBAM(self.embed_dim * 4) if opt.num_levels >= 3 else nn.Identity()
            self.CASA4 = CBAM(self.embed_dim * 4) if opt.num_levels >= 4 else nn.Identity()

        self.ConvsOut = nn.ModuleList([
            SAM(self.embed_dim * 4, in_nc=opt.num_ch) if opt.num_levels >= 4 else nn.Identity(),
            SAM(self.embed_dim * 4, in_nc=opt.num_ch) if opt.num_levels >= 3 else nn.Identity(),
            SAM(self.embed_dim * 2, in_nc=opt.num_ch) if opt.num_levels >= 2 else nn.Identity(),
            ])

        cat_ch = {1: 1, 2: 3, 3: 7, 4: 11}[opt.num_levels]

        self.MIBs = nn.ModuleList([
            MIB(self.embed_dim * cat_ch, self.embed_dim * 1)  if opt.num_levels >= 1 else nn.Identity(),
            MIB1(self.embed_dim * cat_ch, self.embed_dim * 2) if opt.num_levels >= 2 else nn.Identity(),
            MIB1(self.embed_dim * cat_ch, self.embed_dim * 4) if opt.num_levels >= 3 else nn.Identity(),
            MIB1(self.embed_dim * cat_ch, self.embed_dim * 4) if opt.num_levels >= 4 else nn.Identity(),
        ])

        self.Convs = nn.ModuleList([
            BasicConv(self.embed_dim * 4 * 2, self.embed_dim * 4, kernel_size=1, relu=True, stride=1)  if opt.num_levels >= 4 else nn.Identity(),
            BasicConv(self.embed_dim * 2 * 2, self.embed_dim * 2, kernel_size=1, relu=True, stride=1)  if opt.num_levels >= 3 else nn.Identity(),
            BasicConv(self.embed_dim * 2, self.embed_dim, kernel_size=1, relu=True, stride=1)  if opt.num_levels >= 2 else nn.Identity(),
        ])


    def forward(self, x, validation=False):

        if validation:
            x = x.squeeze(dim=0)

        if len(x.shape) == 3:
            x = x.unsqueeze(dim=0)

        x2 = F.interpolate(x, scale_factor=0.5) if self.opt.num_levels >= 2 else None # 1, 4, 128, 128
        x3 = F.interpolate(x2, scale_factor=0.5) if self.opt.num_levels >= 3 else None # 1, 4, 64, 64
        x4 = F.interpolate(x3, scale_factor=0.5) if self.opt.num_levels >= 4 else None # 1, 4, 32, 32

        z2 = checkpoint.checkpoint(self.SCM2, x2, use_reentrant=False) if self.opt.num_levels >= 2 else None # 1, 64, 128, 128
        z3 = checkpoint.checkpoint(self.SCM3, x3, use_reentrant=False) if self.opt.num_levels >= 3 else None # 1, 128, 64, 64
        z4 = checkpoint.checkpoint(self.SCM4, x4, use_reentrant=False) if self.opt.num_levels >= 4 else None # 1, 128, 32, 32

        outputs = list()

        ''' Level == 1 '''
        z = checkpoint.checkpoint(self.FEM[0], x, use_reentrant=False) # 1, 32, 256, 256

        for layer in (self.Encoder[0] if isinstance(self.Encoder[0], nn.Sequential) else [self.Encoder[0]]):#self.Encoder[0].layers):
            z = checkpoint.checkpoint(layer, z, use_reentrant=False)
        res1 = z

        ''' Level >= 2 '''
        if self.opt.num_levels >= 2:
            z = checkpoint.checkpoint(self.FEM[1], res1, use_reentrant=False) # 1, 64, 128, 128
            z = checkpoint.checkpoint(self.FAM2, z, z2, use_reentrant=False) # 1, 64, 128, 128
            for layer in (self.Encoder[1] if isinstance(self.Encoder[1], nn.Sequential) else [self.Encoder[1]]):#self.Encoder[1].layers):
                z = checkpoint.checkpoint(layer, z, use_reentrant=False)
            res2 = z

        ''' Level >= 3 '''
        if self.opt.num_levels >= 3:
            z = checkpoint.checkpoint(self.FEM[2], res2, use_reentrant=False) # 1, 128, 64, 64
            z = checkpoint.checkpoint(self.FAM3, z, z3, use_reentrant=False)
            for layer in (self.Encoder[2] if isinstance(self.Encoder[2], nn.Sequential) else [self.Encoder[2]]):#self.Encoder[2].layers):
                z = checkpoint.checkpoint(layer, z, use_reentrant=False)
            res3 = z

        ''' Level >= 4 '''
        if self.opt.num_levels >= 4:
            z = checkpoint.checkpoint(self.FEM[3], res3, use_reentrant=False) # 1, 128, 32, 32
            z = checkpoint.checkpoint(self.FAM4, z, z4, use_reentrant=False) # 1, 128, 32, 32
            for layer in (self.Encoder[3] if isinstance(self.Encoder[3], nn.Sequential) else [self.Encoder[3]]):#self.Encoder[3].layers):
                z = checkpoint.checkpoint(layer, z, use_reentrant=False)

        z12 = F.interpolate(res1, scale_factor=0.5) if self.opt.num_levels >= 2 else None
        z13 = F.interpolate(res1, scale_factor=0.25) if self.opt.num_levels >= 3 else None
        z14 = F.interpolate(res1, scale_factor=0.125) if self.opt.num_levels >= 4 else None

        z21 = F.interpolate(res2, scale_factor=2) if self.opt.num_levels >= 2 else None
        z23 = F.interpolate(res2, scale_factor=0.5) if self.opt.num_levels >= 3 else None
        z24 = F.interpolate(res2, scale_factor=0.25) if self.opt.num_levels >= 4 else None

        z34 = F.interpolate(res3, scale_factor=0.5) if self.opt.num_levels >= 4 else None
        z32 = F.interpolate(res3, scale_factor=2) if self.opt.num_levels >= 3 else None
        z31 = F.interpolate(z32, scale_factor=2) if self.opt.num_levels >= 3 else None

        z43 = F.interpolate(z, scale_factor=2) if self.opt.num_levels >= 4 else None
        z42 = F.interpolate(z43, scale_factor=2) if self.opt.num_levels >= 4 else None
        z41 = F.interpolate(z42, scale_factor=2) if self.opt.num_levels >= 4 else None

        ### AMIB ###
        z = checkpoint.checkpoint(self.MIBs[3], z14, z24, z34, z, use_reentrant=False) if self.opt.num_levels >= 4 else None # 1, 128, 32, 32
        res3 = checkpoint.checkpoint(self.MIBs[2], z13, z23, res3, z43, use_reentrant=False) if self.opt.num_levels >= 3 else None
        res2 = checkpoint.checkpoint(self.MIBs[1], z12, res2, z32, z42, use_reentrant=False) if self.opt.num_levels >= 2 else None
        res1 = checkpoint.checkpoint(self.MIBs[0], res1, z21, z31, z41, use_reentrant=False)

        if self.opt.apply_CASA:
            z = checkpoint.checkpoint(self.CASA4, z, use_reentrant=False) if self.opt.num_levels >= 4 else None
            res3 = checkpoint.checkpoint(self.CASA3, res3, use_reentrant=False) if self.opt.num_levels >= 3 else None
            res2 = checkpoint.checkpoint(self.CASA2, res2, use_reentrant=False) if self.opt.num_levels >= 2 else None
            res1 = checkpoint.checkpoint(self.CASA1, res1, use_reentrant=False)
        ### AMIB ###

        if self.opt.num_levels >= 4:
            for layer in (self.Decoder[0] if isinstance(self.Decoder[0], nn.Sequential) else [self.Decoder[0]]):#self.Decoder[0].layers):
                z = checkpoint.checkpoint(layer, z, use_reentrant=False)
            z, z_ = checkpoint.checkpoint(self.ConvsOut[0], z, x4, use_reentrant=False) # 1, 128, 32, 32 / 1, 4, 32, 32
            z = checkpoint.checkpoint(self.FEM[4], z, use_reentrant=False) # 1, 128, 64, 64
            outputs.append(z_)

        if self.opt.num_levels >= 3:
            z = torch.cat([z, res3], dim=1) if self.opt.num_levels >= 4 else res3
            z = checkpoint.checkpoint(self.Convs[0], z, use_reentrant=False) if self.opt.num_levels >= 4 else z # 1, 128, 64, 64
            for layer in (self.Decoder[1] if isinstance(self.Decoder[1], nn.Sequential) else [self.Decoder[1]]):#self.Decoder[1].layers):
                z = checkpoint.checkpoint(layer, z, use_reentrant=False)
            z, z_ = checkpoint.checkpoint(self.ConvsOut[1], z, x3, use_reentrant=False)
            z = checkpoint.checkpoint(self.FEM[5], z, use_reentrant=False) # 1, 64, 128, 128
            outputs.append(z_)

        if self.opt.num_levels >= 2:
            z = torch.cat([z, res2], dim=1) if self.opt.num_levels >= 3 else res2 # 1, 128, 128, 128
            z = checkpoint.checkpoint(self.Convs[1], z, use_reentrant=False) if self.opt.num_levels >= 3 else z # 1, 64, 128, 128
            for layer in (self.Decoder[2] if isinstance(self.Decoder[2], nn.Sequential) else [self.Decoder[2]]):#self.Decoder[2].layers):
                z = checkpoint.checkpoint(layer, z, use_reentrant=False)
            z, z_ = checkpoint.checkpoint(self.ConvsOut[2], z, x2, use_reentrant=False) # 1, 64, 128, 128 / 1, 4, 128, 128
            z = checkpoint.checkpoint(self.FEM[6], z, use_reentrant=False) # 1, 32, 256, 256
            outputs.append(z_)

        z = torch.cat([z, res1], dim=1) if self.opt.num_levels >= 2 else res1 # 1, 64, 256, 256
        z = checkpoint.checkpoint(self.Convs[2], z, use_reentrant=False) if self.opt.num_levels >= 2 else z # 1, 32, 256, 256
        for layer in (self.Decoder[3] if isinstance(self.Decoder[3], nn.Sequential) else [self.Decoder[3]]):#self.Decoder[3].layers):
            z = checkpoint.checkpoint(layer, z, use_reentrant=False)
        z = checkpoint.checkpoint(self.FEM[7], z, use_reentrant=False) # 1, 4, 256, 256

        outputs.append(z + x)

        if validation:
            outputs = z + x

        return outputs
