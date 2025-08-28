import torch
import torch.nn as nn
import numpy as np
import math


# -------------Initialization----------------------------------------
def init_weights(*modules):
    for module in modules:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                # print("nn.Conv2D is initialized by variance_scaling_initializer")
                variance_scaling_initializer(m.weight)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

def variance_scaling_initializer(tensor):
    from scipy.stats import truncnorm
    def calculate_fan(shape, factor=2.0, mode='FAN_IN', uniform=False):
        # 64 9 3 3 -> 3 3 9 64
        # 64 64 3 3 -> 3 3 64 64
        if shape:
            # fan_in = float(shape[1]) if len(shape) > 1 else float(shape[0])
            # fan_out = float(shape[0])
            fan_in = float(shape[-2]) if len(shape) > 1 else float(shape[-1])
            fan_out = float(shape[-1])
        else:
            fan_in = 1.0
            fan_out = 1.0
        for dim in shape[:-2]:
            fan_in *= float(dim)
            fan_out *= float(dim)
        if mode == 'FAN_IN':
            # Count only number of input connections.
            n = fan_in
        elif mode == 'FAN_OUT':
            # Count only number of output connections.
            n = fan_out
        elif mode == 'FAN_AVG':
            # Average number of inputs and output connections.
            n = (fan_in + fan_out) / 2.0
        if uniform:
            raise NotImplemented
            # # To get stddev = math.sqrt(factor / n) need to adjust for uniform.
            # limit = math.sqrt(3.0 * factor / n)
            # return random_ops.random_uniform(shape, -limit, limit,
            #                                  dtype, seed=seed)
        else:
            # To get stddev = math.sqrt(factor / n) need to adjust for truncated.
            trunc_stddev = math.sqrt(1.3 * factor / n)
        return fan_in, fan_out, trunc_stddev


############################################################
# FusionNet
############################################################

# -------------ResNet Block (One)----------------------------------------
class Resblock(nn.Module):
    def __init__(self):
        super(Resblock, self).__init__()

        channel = 32
        self.conv20 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1,
                                bias=True)
        self.conv21 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1,
                                bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):  # x= hp of ms; y = hp of pan
        rs1 = self.relu(self.conv20(x))  # Bsx32x64x64
        rs1 = self.conv21(rs1)  # Bsx32x64x64
        rs = torch.add(x, rs1)  # Bsx32x64x64
        return rs


class FusionNet(nn.Module):
    def __init__(self, spectral_num, channel=32):
        super(FusionNet, self).__init__()
        # ConvTranspose2d: output = (input - 1)*stride + outpading - 2*padding + kernelsize
        self.spectral_num = spectral_num

        self.conv1 = nn.Conv2d(in_channels=spectral_num, out_channels=channel, kernel_size=3, stride=1, padding=1,
                               bias=True)
        self.res1 = Resblock()
        self.res2 = Resblock()
        self.res3 = Resblock()
        self.res4 = Resblock()

        self.conv3 = nn.Conv2d(in_channels=channel, out_channels=spectral_num, kernel_size=3, stride=1, padding=1,
                               bias=True)

        self.relu = nn.ReLU(inplace=True)

        self.backbone = nn.Sequential(  # method 2: 4 resnet repeated blocks
            self.res1,
            self.res2,
            self.res3,
            self.res4
        )

        # init_weights(self.backbone, self.conv1, self.conv3)   # state initialization, important!
        self.apply(init_weights)

    def forward(self, x, y):  # x= lms; y = pan

        pan_concat = y.repeat(1, self.spectral_num, 1, 1)  # Bsx8x64x64
        input = torch.sub(pan_concat, x)  # Bsx8x64x64
        rs = self.relu(self.conv1(input))  # Bsx32x64x64

        rs = self.backbone(rs)  # ResNet's backbone!
        output = self.conv3(rs)  # Bsx8x64x64

        return output  # lms + outs

class FusionNet_ZS(nn.Module):
    def __init__(self, spectral_num, channel=32):
        super(FusionNet_ZS, self).__init__()
        # ConvTranspose2d: output = (input - 1)*stride + outpading - 2*padding + kernelsize
        self.spectral_num = spectral_num

        self.conv1 = nn.Conv2d(in_channels=spectral_num, out_channels=channel, kernel_size=3, stride=1, padding=1,
                               bias=True)
        self.res1 = Resblock()
        # self.res2 = Resblock()
        # self.res3 = Resblock()
        # self.res4 = Resblock()

        self.conv3 = nn.Conv2d(in_channels=channel, out_channels=spectral_num, kernel_size=3, stride=1, padding=1,
                               bias=True)

        self.relu = nn.ReLU(inplace=True)

        self.backbone = nn.Sequential(  # method 2: 4 resnet repeated blocks
            self.res1,
            # self.res2,
            # self.res3,
            # self.res4
        )

        # init_weights(self.backbone, self.conv1, self.conv3)   # state initialization, important!
        self.apply(init_weights)

    def forward(self, x, y):  # x= lms; y = pan

        pan_concat = y.repeat(1, self.spectral_num, 1, 1)  # Bsx8x64x64
        input = torch.sub(pan_concat, x)  # Bsx8x64x64
        rs = self.relu(self.conv1(input))  # Bsx32x64x64

        rs = self.backbone(rs)  # ResNet's backbone!
        output = self.conv3(rs)  # Bsx8x64x64

        return output  # lms + outs



class FusionNet2(nn.Module):
    def __init__(self, spectral_num, channel=32):
        super(FusionNet2, self).__init__()
        # ConvTranspose2d: output = (input - 1)*stride + outpading - 2*padding + kernelsize
        self.spectral_num = spectral_num
        self.deconv = nn.ConvTranspose2d(in_channels=spectral_num, out_channels=spectral_num, kernel_size=8, stride=4,
                                         padding=2, bias=True)
        self.conv1 = nn.Conv2d(in_channels=spectral_num, out_channels=channel, kernel_size=3, stride=1, padding=1,
                               bias=True)
        self.res1 = Resblock()
        self.res2 = Resblock()
        self.res3 = Resblock()
        self.res4 = Resblock()

        self.conv3 = nn.Conv2d(in_channels=channel, out_channels=spectral_num, kernel_size=3, stride=1, padding=1,
                               bias=True)

        self.relu = nn.ReLU(inplace=True)

        self.backbone = nn.Sequential(  # method 2: 4 resnet repeated blocks
            self.res1,
            self.res2,
            self.res3,
            self.res4
        )

        # init_weights(self.backbone, self.conv1, self.conv3)   # state initialization, important!
        self.apply(init_weights)

    def forward(self, x, y):  # x= lms; y = pan

        pan_concat = y.repeat(1, self.spectral_num, 1, 1)  # Bsx8x64x64
        x = self.deconv(x)
        input = torch.sub(pan_concat, x)  # Bsx8x64x64
        rs = self.relu(self.conv1(input))  # Bsx32x64x64

        rs = self.backbone(rs)  # ResNet's backbone!
        output = self.conv3(rs)  # Bsx8x64x64

        return output  # lms + outs


############################################################
# APNN
############################################################
class APNN(nn.Module):
    def __init__(self, spectral_num):
        super(APNN, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=spectral_num+1, out_channels=48, kernel_size=9, stride=1, padding=4)
        self.conv_2 = nn.Conv2d(in_channels=48, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.conv_3 = nn.Conv2d(in_channels=32, out_channels=spectral_num, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()

    def forward(self, lrms, pan):  # x= lms; y = pan
        input = torch.cat([lrms, pan], dim=1)
        fea = self.relu(self.conv_1(input))  # [batch_size,5,128,128]
        fea = self.relu(self.conv_2(fea))
        out = self.conv_3(fea)
        # out = out + lrms
        return out



class APNN2(nn.Module):
    def __init__(self, spectral_num):
        super(APNN2, self).__init__()
        self.layer_0 = nn.Sequential(
                nn.ConvTranspose2d(in_channels=spectral_num, out_channels=spectral_num, kernel_size=8, stride=4, padding=2, output_padding=0)
            )

        self.conv_1 = nn.Conv2d(in_channels=spectral_num+1, out_channels=48, kernel_size=9, stride=1, padding=4)
        self.bn1 = nn.BatchNorm2d(48) 
        self.conv_2 = nn.Conv2d(in_channels=48, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(32) 
        self.conv_3 = nn.Conv2d(in_channels=32, out_channels=spectral_num, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()

    def forward(self, ms, pan):  # x= lms; y = pan

        lrms = self.layer_0(ms)

        input = torch.cat([lrms, pan], dim=1)
        fea = self.conv_1(input)
        fea = self.bn1(fea)
        fea = self.relu(fea)  # [batch_size,5,128,128]
        fea = self.relu(self.bn2(self.conv_2(fea)))
        out = self.conv_3(fea)
        # out = out + lrms
        return out



############################################################
# MSDCNN
############################################################

class MSDCNN(nn.Module):
    def __init__(self, spectral_num):
        super(MSDCNN, self).__init__()


        input_channel = spectral_num + 1
        output_channel = spectral_num

        self.conv1 = nn.Conv2d(in_channels=input_channel, out_channels=60, kernel_size=7, stride=1, padding=3, bias=True)

        self.conv2_1 = nn.Conv2d(in_channels=60, out_channels=20, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2_2 = nn.Conv2d(in_channels=60, out_channels=20, kernel_size=5, stride=1, padding=2, bias=True)
        self.conv2_3 = nn.Conv2d(in_channels=60, out_channels=20, kernel_size=7, stride=1, padding=3, bias=True)

        self.conv3 = nn.Conv2d(in_channels=60, out_channels=30, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv4_1 = nn.Conv2d(in_channels=30, out_channels=10, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv4_2 = nn.Conv2d(in_channels=30, out_channels=10, kernel_size=5, stride=1, padding=2, bias=True)
        self.conv4_3 = nn.Conv2d(in_channels=30, out_channels=10, kernel_size=7, stride=1, padding=3, bias=True)

        self.conv5 = nn.Conv2d(in_channels=30, out_channels=output_channel, kernel_size=5, stride=1, padding=2, bias=True)

        self.shallow1 = nn.Conv2d(in_channels=input_channel, out_channels=64, kernel_size=9, stride=1, padding=4, bias=True)
        self.shallow2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0, bias=True)
        self.shallow3 = nn.Conv2d(in_channels=32, out_channels=output_channel, kernel_size=5, stride=1, padding=2, bias=True)

        self.relu = nn.ReLU(inplace=True)


    def forward(self, x, y):  # x: lms; y: pan

        concat = torch.cat([x, y], 1)  # Bsx9x64x64

        out1 = self.relu(self.conv1(concat))  # Bsx60x64x64
        out21 = self.conv2_1(out1)   # Bsx20x64x64
        out22 = self.conv2_2(out1)   # Bsx20x64x64
        out23 = self.conv2_3(out1)   # Bsx20x64x64
        out2 = torch.cat([out21, out22, out23], 1)  # Bsx60x64x64

        out2 = self.relu(torch.add(out2, out1))  # Bsx60x64x64

        out3 = self.relu(self.conv3(out2))  # Bsx30x64x64
        out41 = self.conv4_1(out3)          # Bsx10x64x64
        out42 = self.conv4_2(out3)          # Bsx10x64x64
        out43 = self.conv4_3(out3)          # Bsx10x64x64
        out4 = torch.cat([out41, out42, out43], 1)  # Bsx30x64x64

        out4 = self.relu(torch.add(out4, out3))  # Bsx30x64x64

        out5 = self.conv5(out4)  # Bsx8x64x64

        shallow1 = self.relu(self.shallow1(concat))   # Bsx64x64x64
        shallow2 = self.relu(self.shallow2(shallow1))  # Bsx32x64x64
        shallow3 = self.shallow3(shallow2) # Bsx8x64x64

        out = torch.add(out5, shallow3)  # Bsx8x64x64
        out = self.relu(out)  # Bsx8x64x64

        return out

############################################################
# DRPNN
############################################################


class Repeatblock(nn.Module):
    def __init__(self):
        super(Repeatblock, self).__init__()

        channel = 32  # input_channel =
        self.conv2 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=7, stride=1, padding=3,
                               bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        rs = self.relu(self.conv2(x))

        return rs

class DRPNN(nn.Module):
    def __init__(self, spectral_num, channel=32):
        super(DRPNN, self).__init__()

        # ConvTranspose2d: output = (input - 1)*stride + outpading - 2*padding + kernelsize
        self.conv1 = nn.Conv2d(in_channels=spectral_num+1, out_channels=channel, kernel_size=7, stride=1, padding=3,
                                  bias=True)

        self.conv2 = nn.Conv2d(in_channels=channel, out_channels=spectral_num+1, kernel_size=7, stride=1, padding=3,
                                  bias=True)
        self.conv3 = nn.Conv2d(in_channels=spectral_num+1, out_channels=spectral_num, kernel_size=7, stride=1, padding=3,
                                  bias=True)
        self.relu = nn.ReLU(inplace=True)

        self.backbone = nn.Sequential(  # method 2: 4 resnet repeated blocks
            Repeatblock(),
            Repeatblock(),
            Repeatblock(),
            Repeatblock(),
            Repeatblock(),
            Repeatblock(),
            Repeatblock(),
            Repeatblock(),
        )

    def forward(self, x, y):  # x= lms; y = pan

        input = torch.cat([x, y], 1)  # Bsx9x64x64
        rs = self.relu(self.conv1(input))  # Bsx64x64x64

        rs = self.backbone(rs)  # backbone!  Bsx64x64x64

        out_res = self.conv2(rs)  # Bsx9x64x64
        output1 = torch.add(input, out_res)  # Bsx9x64x64
        output  = self.conv3(output1)  # Bsx8x64x64

        return output

############################################################
# DiCNN
############################################################

class DiCNN(nn.Module):
    def __init__(self, spectral_num, channel=64, reg=True):
        super(DiCNN, self).__init__()

        self.reg = reg
        # ConvTranspose2d: output = (input - 1)*stride + outpading - 2*padding + kernelsize
        self.conv1 = nn.Conv2d(in_channels=spectral_num + 1, out_channels=channel, kernel_size=3, stride=1, padding=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=channel, out_channels=spectral_num, kernel_size=3, stride=1, padding=1,
                               bias=True)

        self.relu = nn.ReLU(inplace=True)

        self.apply(init_weights)

    def forward(self, x, y):
        # x= lms; y = pan
        input1 = torch.cat([x, y], 1)  # Bsx9x64x64

        rs = self.relu(self.conv1(input1))
        rs = self.relu(self.conv2(rs))
        out = self.conv3(rs)
        output = x + out

        return output

############################################################
# Pan-Mamba
############################################################   

#
# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from einops import rearrange
# import numbers
# from mamba_ssm.modules.mamba_simple import Mamba
# # from mamba_simple import Mamba
# # from .refine import Refine
#
#
# class Refine(nn.Module):
#
#     def __init__(self, n_feat, out_channel):
#         super(Refine, self).__init__()
#
#         self.conv_in = nn.Conv2d(n_feat, n_feat, 3, stride=1, padding=1)
#         self.process = nn.Sequential(
#             # CALayer(n_feat,4),
#             # CALayer(n_feat,4),
#             ChannelAttention(n_feat, 4))
#         self.conv_last = nn.Conv2d(in_channels=n_feat, out_channels=out_channel, kernel_size=3, stride=1, padding=1)
#
#     def forward(self, x):
#         out = self.conv_in(x)
#         out = self.process(out)
#         out = self.conv_last(out)
#
#         return out
#
# class ChannelAttention(nn.Module):
#     def __init__(self, channel, reduction):
#         super(ChannelAttention, self).__init__()
#         # global average pooling: feature --> point
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         # feature channel downscale and upscale --> channel weight
#         self.conv_du = nn.Sequential(
#             nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
#             nn.Sigmoid()
#         )
#         self.process = nn.Sequential(
#             nn.Conv2d(channel, channel, 3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(channel, channel, 3, stride=1, padding=1)
#         )
#
#     def forward(self, x):
#         res = self.process(x)
#         y = self.avg_pool(res)
#         z = self.conv_du(y)
#         return z *res + x
#
# def to_3d(x):
#     return rearrange(x, 'b c h w -> b (h w) c')
#
# class FeedForward(nn.Module):
#     def __init__(self, dim, ffn_expansion_factor, bias):
#         super(FeedForward, self).__init__()
#
#         hidden_features = int(dim*ffn_expansion_factor)
#
#         self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
#
#         self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
#
#         self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
#
#     def forward(self, x):
#         x = self.project_in(x)
#         x1, x2 = self.dwconv(x).chunk(2, dim=1)
#         x = F.gelu(x1) * x2
#         x = self.project_out(x)
#         return x
#
# class CrossAttention(nn.Module):
#     def __init__(self, dim, num_heads, bias):
#         super(CrossAttention, self).__init__()
#         self.num_heads = num_heads
#         self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
#
#         self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
#         self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
#         self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
#         self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=bias)
#         self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
#
#     def forward(self, ms, pan):
#         b, c, h, w = ms.shape
#
#         kv = self.kv_dwconv(self.kv(pan))
#         k, v = kv.chunk(2, dim=1)
#         q = self.q_dwconv(self.q(ms))
#
#         q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#         k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#         v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#
#         q = torch.nn.functional.normalize(q, dim=-1)
#         k = torch.nn.functional.normalize(k, dim=-1)
#
#         attn = (q @ k.transpose(-2, -1)) * self.temperature
#         attn = attn.softmax(dim=-1)
#
#         out = (attn @ v)
#
#         out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
#
#         out = self.project_out(out)
#         return out
#
# def to_3d(x):
#     return rearrange(x, 'b c h w -> b (h w) c')
#
# class FeedForward(nn.Module):
#     def __init__(self, dim, ffn_expansion_factor, bias):
#         super(FeedForward, self).__init__()
#
#         hidden_features = int(dim*ffn_expansion_factor)
#
#         self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
#
#         self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
#
#         self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
#
#     def forward(self, x):
#         x = self.project_in(x)
#         x1, x2 = self.dwconv(x).chunk(2, dim=1)
#         x = F.gelu(x1) * x2
#         x = self.project_out(x)
#         return x
# class CrossAttention(nn.Module):
#     def __init__(self, dim, num_heads, bias):
#         super(CrossAttention, self).__init__()
#         self.num_heads = num_heads
#         self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
#
#         self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
#         self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
#         self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
#         self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=bias)
#         self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
#
#     def forward(self, ms, pan):
#         b, c, h, w = ms.shape
#
#         kv = self.kv_dwconv(self.kv(pan))
#         k, v = kv.chunk(2, dim=1)
#         q = self.q_dwconv(self.q(ms))
#
#         q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#         k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#         v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#
#         q = torch.nn.functional.normalize(q, dim=-1)
#         k = torch.nn.functional.normalize(k, dim=-1)
#
#         attn = (q @ k.transpose(-2, -1)) * self.temperature
#         attn = attn.softmax(dim=-1)
#
#         out = (attn @ v)
#
#         out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
#
#         out = self.project_out(out)
#         return out
#
# def to_4d(x, h, w):
#     return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
# class TransformerBlock(nn.Module):
#     def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
#         super(TransformerBlock, self).__init__()
#         self.norm_cro1= LayerNorm(dim, LayerNorm_type)
#         self.norm_cro2 = LayerNorm(dim, LayerNorm_type)
#         self.norm1 = LayerNorm(dim, LayerNorm_type)
#         self.norm2 = LayerNorm(dim, LayerNorm_type)
#         self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
#         self.cro = CrossAttention(dim,num_heads,bias)
#         self.proj = nn.Conv2d(dim,dim,1,1,0)
#     def forward(self, ms,pan):
#         ms = ms+self.cro(self.norm_cro1(ms),self.norm_cro2(pan))
#         ms = ms + self.ffn(self.norm2(ms))
#         return ms
#
# #对输入 x 进行标准化处理（除以标准差），然后乘以可学习的缩放参数 weight。没有偏置项。
# class BiasFree_LayerNorm(nn.Module):
#     def __init__(self, normalized_shape):
#         super(BiasFree_LayerNorm, self).__init__()
#         if isinstance(normalized_shape, numbers.Integral):
#             normalized_shape = (normalized_shape,)
#         normalized_shape = torch.Size(normalized_shape)
#
#         assert len(normalized_shape) == 1
#
#         self.weight = nn.Parameter(torch.ones(normalized_shape))
#         self.normalized_shape = normalized_shape
#
#     def forward(self, x):
#         sigma = x.var(-1, keepdim=True, unbiased=False)
#         return x / torch.sqrt(sigma+1e-5) * self.weight
#
# #对输入 x 进行标准化处理（减去均值并除以标准差），然后乘以可学习的缩放参数 weight，最后加上可学习的偏置参数 bias
# class WithBias_LayerNorm(nn.Module):
#     def __init__(self, normalized_shape):
#         super(WithBias_LayerNorm, self).__init__()
#         if isinstance(normalized_shape, numbers.Integral):
#             normalized_shape = (normalized_shape,)
#         normalized_shape = torch.Size(normalized_shape)
#
#         assert len(normalized_shape) == 1
#
#         self.weight = nn.Parameter(torch.ones(normalized_shape))
#         self.bias = nn.Parameter(torch.zeros(normalized_shape))
#         self.normalized_shape = normalized_shape
#
#     def forward(self, x):
#         mu = x.mean(-1, keepdim=True)
#         sigma = x.var(-1, keepdim=True, unbiased=False)
#         return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias
#
#
# class LayerNorm(nn.Module):
#     def __init__(self, dim, LayerNorm_type):
#         super(LayerNorm, self).__init__()
#         if LayerNorm_type =='BiasFree':
#             self.body = BiasFree_LayerNorm(dim)
#         else:
#             self.body = WithBias_LayerNorm(dim)
#
#     def forward(self, x):
#         h, w = x.shape[-2:]
#         return to_4d(self.body(to_3d(x)), h, w)
# # ---------------------------------------------------------------------------------------------------------------------
#
# class LayerNorm(nn.Module):
#     def __init__(self, dim, LayerNorm_type):
#         super(LayerNorm, self).__init__()
#         if LayerNorm_type =='BiasFree':
#             self.body = BiasFree_LayerNorm(dim)
#         else:
#             self.body = WithBias_LayerNorm(dim)
#
#     def forward(self, x):
#         if len(x.shape)==4:   # 如果输入是四维的（例如图像），则先展平为三维，再进行归一化，然后恢复为四维。
#             h, w = x.shape[-2:]
#             return to_4d(self.body(to_3d(x)), h, w)
#         else:
#             return self.body(x)
#
# #PatchUnEmbed 类用于将补丁表示恢复为原始图像大小。
# class PatchUnEmbed(nn.Module):
#     def __init__(self,basefilter) -> None:
#         super().__init__()
#         self.nc = basefilter
#     def forward(self, x,x_size):
#         B,HW,C = x.shape
#         x = x.transpose(1, 2).view(B, self.nc, x_size[0], x_size[1])  # B Ph*Pw C
#         return x
#
# #PatchEmbed 类用于将二维图像嵌入到补丁表示中。
# class PatchEmbed(nn.Module):
#     """ 2D Image to Patch Embedding
#     """
#     def __init__(self,patch_size=4, stride=4,in_chans=36, embed_dim=32*32*32, norm_layer=None, flatten=True):
#         super().__init__()
#         # patch_size = to_2tuple(patch_size)
#         self.patch_size = patch_size
#         self.flatten = flatten
#
#         self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
#         self.norm = LayerNorm(embed_dim,'BiasFree')
#
#     def forward(self, x):
#         #（b,c,h,w)->(b,c*s*p,h//s,w//s)
#         #(b,h*w//s**2,c*s**2)
#         B, C, H, W = x.shape
#         # x = F.unfold(x, self.patch_size, stride=self.patch_size)
#         x = self.proj(x)
#         if self.flatten:
#             x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
#         # x = self.norm(x)
#         return x
#
# class SingleMambaBlock(nn.Module):
#     def __init__(self, dim):
#         super(SingleMambaBlock, self).__init__()
#         self.encoder = Mamba(dim,bimamba_type=None)
#         self.norm = LayerNorm(dim,'with_bias')
#         # self.PatchEmbe=PatchEmbed(patch_size=4, stride=4,in_chans=dim, embed_dim=dim*16)
#     def forward(self,ipt):
#         x,residual = ipt
#         residual = x+residual
#         x = self.norm(residual)
#         return (self.encoder(x),residual)
#
# #通过将全色图像的部分特征与多光谱图像的部分特征结合，可以使得后续的编码器（msencoder 和 panencoder）处理更加丰富和多样化的特征
# #pan_first_half 提取了 pan（全色图像特征）的前一半特征。然后使用 torch.cat 函数将 pan_first_half 和 ms 的后一半特征（即 ms[:, :, C // 2:]）沿着特征维度（dim=2）拼接在一起，形成新的特征张量 ms_swap。
# class TokenSwapMamba(nn.Module):
#     def __init__(self, dim):
#         super(TokenSwapMamba, self).__init__()
#         self.msencoder = Mamba(dim,bimamba_type=None)
#         self.panencoder = Mamba(dim,bimamba_type=None)
#         self.norm1 = LayerNorm(dim,'with_bias')
#         self.norm2 = LayerNorm(dim,'with_bias')
#     def forward(self, ms,pan
#                 ,ms_residual,pan_residual):
#         # ms (B,N,C)
#         #pan (B,N,C)
#         ms_residual = ms+ms_residual
#         pan_residual = pan+pan_residual
#         ms = self.norm1(ms_residual)
#         pan = self.norm2(pan_residual)
#         B,N,C = ms.shape
#         ms_first_half = ms[:, :, :C//2]
#         pan_first_half = pan[:, :, :C//2]
#         ms_swap= torch.cat([pan_first_half,ms[:,:,C//2:]],dim=2)
#         pan_swap= torch.cat([ms_first_half,pan[:,:,C//2:]],dim=2)
#         ms_swap = self.msencoder(ms_swap)
#         pan_swap = self.panencoder(pan_swap)
#         return ms_swap,pan_swap,ms_residual,pan_residual
#
# #用于跨模态特征融合
# class CrossMamba(nn.Module):
#     def __init__(self, dim):
#         super(CrossMamba, self).__init__()
#         self.cross_mamba = Mamba(dim,bimamba_type="v3")
#         self.norm1 = LayerNorm(dim,'with_bias')
#         self.norm2 = LayerNorm(dim,'with_bias')
#         self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
#     def forward(self,ms,ms_resi,pan):
#         ms_resi = ms+ms_resi
#         ms = self.norm1(ms_resi)
#         pan = self.norm2(pan)
#         global_f = self.cross_mamba(self.norm1(ms),extra_emb=self.norm2(pan))
#         B,HW,C = global_f.shape
#         S = int(math.sqrt(HW))
#         # ms = global_f.transpose(1, 2).view(B, C, 128*8, 128*8)
#         ms = global_f.transpose(1, 2).view(B, C, S, S)
#         ms =  (self.dwconv(ms)+ms).flatten(2).transpose(1, 2)
#         return ms,ms_resi
#
# #这段代码的设计目的是通过部分应用实例归一化来增强卷积神经网络的表达能力和稳定性。
# #这种设计希望在局部归一化与保留原始特征之间找到一个平衡，从而在提升训练稳定性的同时，不损失特征多样性，最终达到更好的训练效果和模型性能。
# class HinResBlock(nn.Module):
#     def __init__(self, in_size, out_size, relu_slope=0.2, use_HIN=True):
#         super(HinResBlock, self).__init__()
#         self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)
#
#         self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
#         self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
#         self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
#         self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)
#         if use_HIN:
#             # nn.InstanceNorm2d 是 PyTorch 中的一个层归一化模块，专门用于二维图像数据的实例归一化。
#             # 实例归一化是对每个样本的每个通道独立进行归一化处理，这与批归一化的区别在于它不依赖于批次的均值和方差。
#             self.norm = nn.InstanceNorm2d(out_size // 2, affine=True) #affine: 是否学习可训练的缩放和平移参数（默认为 False）
#         self.use_HIN = use_HIN
#
#     def forward(self, x):
#         resi = self.relu_1(self.conv_1(x))
#         #将张量 resi 沿着第一个维度（通常是通道维度）分割成两个子张量
#         out_1, out_2 = torch.chunk(resi, 2, dim=1)
#         #通过对 out_1 应用实例归一化，能够标准化部分特征图，使得这些特征图具有零均值和单位方差，从而提高训练的稳定性
#         #将未归一化的 out_2 与归一化的 out_1 拼接，保留了部分原始特征信息，避免了过度归一化可能带来的信息丢失
#         resi = torch.cat([self.norm(out_1), out_2], dim=1)
#         resi = self.relu_2(self.conv_2(resi))
#         return x+resi
#
# class PanMamba(nn.Module):
#     def __init__(self,num_channels=None,base_filter=None,args=None):
#         super(PanMamba, self).__init__()
#         base_filter=32
#         self.base_filter = base_filter
#         self.stride=1
#         self.patch_size=1
#         self.pan_encoder = nn.Sequential(nn.Conv2d(1,base_filter,3,1,1),HinResBlock(base_filter,base_filter),HinResBlock(base_filter,base_filter),HinResBlock(base_filter,base_filter))
#         self.ms_encoder = nn.Sequential(nn.Conv2d(num_channels,base_filter,3,1,1),HinResBlock(base_filter,base_filter),HinResBlock(base_filter,base_filter),HinResBlock(base_filter,base_filter))
#         self.embed_dim = base_filter*self.stride*self.patch_size
#         self.shallow_fusion1 = nn.Conv2d(base_filter*2,base_filter,3,1,1)
#         self.shallow_fusion2 = nn.Conv2d(base_filter*2,base_filter,3,1,1)
#         self.ms_to_token = PatchEmbed(in_chans=base_filter,embed_dim=self.embed_dim,patch_size=self.patch_size,stride=self.stride)
#         self.pan_to_token = PatchEmbed(in_chans=base_filter,embed_dim=self.embed_dim,patch_size=self.patch_size,stride=self.stride)
#         self.deep_fusion1= CrossMamba(self.embed_dim)
#         self.deep_fusion2 = CrossMamba(self.embed_dim)
#         self.deep_fusion3 = CrossMamba(self.embed_dim)
#         self.deep_fusion4 = CrossMamba(self.embed_dim)
#         self.deep_fusion5 = CrossMamba(self.embed_dim)
#
#         self.pan_feature_extraction = nn.Sequential(*[SingleMambaBlock(self.embed_dim) for i in range(8)])
#         self.ms_feature_extraction = nn.Sequential(*[SingleMambaBlock(self.embed_dim) for i in range(8)])
#         self.swap_mamba1 = TokenSwapMamba(self.embed_dim)
#         self.swap_mamba2 = TokenSwapMamba(self.embed_dim)
#         self.patchunembe = PatchUnEmbed(base_filter)
#         self.output = Refine(base_filter,num_channels)
#
#     # def forward(self,ms,_,pan):
#     def forward(self,ms,pan):
#
#         ms_bic = F.interpolate(ms,scale_factor=4)
#         ms_f = self.ms_encoder(ms_bic)
#         # ms_f = ms_bic
#         # pan_f = pan
#         b,c,h,w = ms_f.shape
#         pan_f = self.pan_encoder(pan)
#         ms_f = self.ms_to_token(ms_f)
#         pan_f = self.pan_to_token(pan_f)
#         residual_ms_f = 0
#         residual_pan_f = 0
#         ms_f,residual_ms_f = self.ms_feature_extraction([ms_f,residual_ms_f])
#         pan_f,residual_pan_f = self.pan_feature_extraction([pan_f,residual_pan_f])
#         ms_f,pan_f,residual_ms_f,residual_pan_f = self.swap_mamba1(ms_f,pan_f,residual_ms_f,residual_pan_f)
#         ms_f,pan_f,residual_ms_f,residual_pan_f = self.swap_mamba2(ms_f,pan_f,residual_ms_f,residual_pan_f)
#         ms_f = self.patchunembe(ms_f,(h,w))
#         pan_f = self.patchunembe(pan_f,(h,w))
#         ms_f = self.shallow_fusion1(torch.concat([ms_f,pan_f],dim=1))+ms_f
#         pan_f = self.shallow_fusion2(torch.concat([pan_f,ms_f],dim=1))+pan_f
#         ms_f = self.ms_to_token(ms_f)
#         pan_f = self.pan_to_token(pan_f)
#         residual_ms_f = 0
#         ms_f,residual_ms_f = self.deep_fusion1(ms_f,residual_ms_f,pan_f)
#         ms_f,residual_ms_f = self.deep_fusion2(ms_f,residual_ms_f,pan_f)
#         ms_f,residual_ms_f = self.deep_fusion3(ms_f,residual_ms_f,pan_f)
#         ms_f,residual_ms_f = self.deep_fusion4(ms_f,residual_ms_f,pan_f)
#         ms_f,residual_ms_f = self.deep_fusion5(ms_f,residual_ms_f,pan_f)
#         ms_f = self.patchunembe(ms_f,(h,w))
#         hrms = self.output(ms_f)+ms_bic
#         return hrms


############################################################
# CANNet
############################################################    
# from canconv.layers.canconv import CANConv
#
# class CANResBlock(nn.Module):
#     def __init__(self, channels, cluster_num, filter_threshold, cluster_source="channel", *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#         self.conv1 = CANConv(channels, channels, cluster_num=cluster_num,
#                             cluster_source=cluster_source, filter_threshold=filter_threshold)
#         self.conv2 = CANConv(channels, channels, cluster_num=cluster_num, filter_threshold=filter_threshold)
#         self.act = nn.LeakyReLU(inplace=True)
#
#     def forward(self, x, cache_indice=None, cluster_override=None):
#         res, idx = self.conv1(x, cache_indice, cluster_override)
#         res = F.leaky_relu(res)
#         res, _ = self.conv2(res, cache_indice, idx)
#         x = x + res
#         return x, idx
#
#
# class ConvDown(nn.Module):
#     def __init__(self, in_channels, dsconv=True, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#
#         if dsconv:
#             self.conv = nn.Sequential(
#                 # nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1),
#                 nn.Conv2d(in_channels, in_channels, 2, 2, 0),
#                 nn.LeakyReLU(inplace=True),
#                 nn.Conv2d(in_channels, in_channels, 3, 1, 1,
#                           groups=in_channels, bias=False),
#                 nn.Conv2d(in_channels, in_channels*2, 1, 1, 0)
#             )
#         else:
#             self.conv = nn.Sequential(
#                 nn.Conv2d(in_channels, in_channels,
#                           kernel_size=3, stride=2, padding=1),
#                 # nn.Conv2d(in_channels, in_channels, 2, 2, 0),
#                 nn.LeakyReLU(inplace=True),
#                 nn.Conv2d(in_channels, in_channels*2, 3, 1, 1)
#             )
#
#     def forward(self, x):
#         return self.conv(x)
#
#
# class ConvUp(nn.Module):
#     def __init__(self, in_channels, dsconv=True, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#
#         # self.conv1 = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=3, stride=2, padding=1, output_padding=1)
#         self.conv1 = nn.ConvTranspose2d(in_channels, in_channels//2, 2, 2, 0)
#         if dsconv:
#             self.conv2 = nn.Sequential(
#                 nn.Conv2d(in_channels//2, in_channels//2, 3, 1,
#                           1, groups=in_channels//2, bias=False),
#                 nn.Conv2d(in_channels//2, in_channels//2, 1, 1, 0)
#             )
#         else:
#             self.conv2 = nn.Conv2d(in_channels//2, in_channels//2, 3, 1, 1)
#
#     def forward(self, x, y):
#         x = F.leaky_relu(self.conv1(x))
#         x = x + y
#         x = F.leaky_relu(self.conv2(x))
#         return x
#
#
#
# class CANNet(nn.Module):
#     def __init__(self, spectral_num=8, channels=32, cluster_num=32, filter_threshold=0.005, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#         self.head_conv = nn.Conv2d(spectral_num+1, channels, 3, 1, 1)
#         self.rb1 = CANResBlock(channels, cluster_num, filter_threshold)
#         self.down1 = ConvDown(channels)
#         self.rb2 = CANResBlock(channels*2, cluster_num, filter_threshold)
#         self.down2 = ConvDown(channels*2)
#         self.rb3 = CANResBlock(channels*4, cluster_num, filter_threshold)
#         self.up1 = ConvUp(channels*4)
#         self.rb4 = CANResBlock(channels*2, cluster_num, filter_threshold)
#         self.up2 = ConvUp(channels*2)
#         self.rb5 = CANResBlock(channels, cluster_num, filter_threshold)
#         self.tail_conv = nn.Conv2d(channels, spectral_num, 3, 1, 1)
#
#     #cache_indice的含义
#     def forward(self, lms, pan, cache_indice=None):
#         x1 = torch.cat([pan, lms], dim=1)
#         x1 = self.head_conv(x1)   # [16, 32, 128, 128]
#         x1, idx1 = self.rb1(x1, cache_indice)
#         x2 = self.down1(x1)
#         x2, idx2 = self.rb2(x2, cache_indice)
#         x3 = self.down2(x2)
#         x3, _ = self.rb3(x3, cache_indice)
#         x4 = self.up1(x3, x2)
#         del x2
#         x4, _ = self.rb4(x4, cache_indice, idx2)
#         del idx2
#         x5 = self.up2(x4, x1)
#         del x1
#         x5, _ = self.rb5(x5, cache_indice, idx1)
#         del idx1
#         x5 = self.tail_conv(x5)
#         # return lms + x5
#         return x5
#
#
# ############################################################
# # CANFusionNet  将FusionNet的中间两层残差模块换成了CANCRB模块
# ############################################################
# class CANCRB(nn.Module):
#     def __init__(self, in_planes, cluster_num=32):
#         super(CANCRB, self).__init__()
#         self.conv1 = CANConv(in_planes, in_planes, cluster_num=cluster_num)
#         self.relu1 = nn.ReLU()
#         self.conv2 = CANConv(in_planes, in_planes, cluster_num=cluster_num)
#
#     def forward(self, x, cache_indice=None):
#         res, idx = self.conv1(x, cache_indice)
#         res = self.relu1(res)
#         res, _ = self.conv2(res, cache_indice, idx)
#         x = x + res
#         return x
#
#
# class CANFusionNet(nn.Module):
#     def __init__(self, spectral_num, channel=32):
#         super(CANFusionNet, self).__init__()
#         # ConvTranspose2d: output = (input - 1)*stride + outpading - 2*padding + kernelsize
#         self.spectral_num = spectral_num
#
#         self.conv1 = nn.Conv2d(in_channels=spectral_num, out_channels=channel, kernel_size=3, stride=1, padding=1,
#                                bias=True)
#         self.res1 = Resblock()
#         self.res2 = CANCRB(channel)
#         self.res3 = CANCRB(channel)
#         self.res4 = Resblock()
#
#         self.conv3 = nn.Conv2d(in_channels=channel, out_channels=spectral_num, kernel_size=3, stride=1, padding=1,
#                                bias=True)
#
#         self.relu = nn.ReLU(inplace=True)
#
#         # init_weights(self.backbone, self.conv1, self.conv3)   # state initialization, important!
#         # self.apply(init_weights)
#
#     def forward(self, lms, pan, idx=None):  # x= lms; y = pan
#
#         pan_concat = pan.repeat(1, self.spectral_num, 1, 1)  # Bsx8x64x64
#         input = torch.sub(pan_concat, lms)  # Bsx8x64x64
#         rs = self.relu(self.conv1(input))  # Bsx32x64x64
#
#         rs = self.res1(rs)
#         rs = self.res2(rs, idx)
#         rs = self.res3(rs, idx)
#         rs = self.res4(rs)
#
#         output = self.conv3(rs)  # Bsx8x64x64
#
#         return output
#
import torch
import torch.nn.functional as F


def get_hp(data):

    rs = F.avg_pool2d(data, kernel_size=5, stride=1, padding=2)
    rs = data - rs
    return rs


class PanNet(nn.Module):
    def __init__(self, spectral_num=4, channel=32):
        super(PanNet, self).__init__()
  
        # ConvTranspose2d: output = (input - 1)*stride + outpading - 2*padding + kernelsize
        self.deconv = nn.ConvTranspose2d(in_channels=spectral_num, out_channels=spectral_num, kernel_size=8, stride=4,
                                         padding=2, bias=True)
        self.conv1 = nn.Conv2d(in_channels=spectral_num + 1, out_channels=channel, kernel_size=3, stride=1, padding=1,
                               bias=True)
        self.res1 = Resblock()
        self.res2 = Resblock()
        self.res3 = Resblock()
        self.res4 = Resblock()

        self.conv3 = nn.Conv2d(in_channels=channel, out_channels=spectral_num, kernel_size=3, stride=1, padding=1,
                               bias=True)

        self.relu = nn.ReLU(inplace=True)

        self.backbone = nn.Sequential(  # method 2: 4 resnet repeated blocks
            self.res1,
            self.res2,
            self.res3,
            self.res4
        )

        # init_weights(self.backbone, self.deconv, self.conv1, self.conv3)  # state initialization, important!
        self.apply(init_weights)
        # init_weights(self.backbone, self.deconv, self.conv1, self.conv3)  # state initialization, important!

    def forward(self, x, y):  # x= lms; y = pan

        x=get_hp(x)
        y=get_hp(y)

        # output_deconv = self.deconv(x)
        input = torch.cat([x, y], 1)  # Bsx9x64x64
        rs = self.relu(self.conv1(input))  # Bsx32x64x64

        rs = self.backbone(rs)  # ResNet's backbone!

        output = self.conv3(rs)  # Bsx8x64x64

        return output


class PNN(nn.Module):
    def __init__(self, spectral_num=4):
        super(PNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=spectral_num+1, out_channels=64, kernel_size=9, stride=1, padding=4)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=spectral_num, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()


    def forward(self, lrms, pan):
        input = torch.cat([lrms, pan], dim=1)  # 64,9,64,64
        fea = self.relu(self.conv1(input))  # [batch_size,5,128,128]
        fea = self.relu(self.conv2(fea))
        out = self.conv3(fea)
        return out


############################################################
# MTF_GLP_HPM
############################################################
from scipy import ndimage
from torch.nn.functional import interpolate
import cv2

def downsample_nearest(imgs, r=4):
    _, __, h, w = imgs.shape
    return interpolate(imgs, size=[h // r, w // r], mode='nearest')

def upsample_bicubic(imgs, r=4):
    _, __, h, w = imgs.shape
    return interpolate(imgs, size=[h * r, w * r], mode='bicubic')

def upsample_interp23(image, ratio=4):  # image: [batch,C,X//4,Y//4]
    image = np.transpose(image, (2, 0, 1))

    b, r, c = image.shape

    CDF23 = 2 * np.array(
        [0.5, 0.305334091185, 0, -0.072698593239, 0, 0.021809577942, 0, -0.005192756653, 0, 0.000807762146, 0,
         -0.000060081482])
    d = CDF23[::-1]
    CDF23 = np.insert(CDF23, 0, d[:-1])
    BaseCoeff = CDF23

    first = 1
    for z in range(1, np.int(np.log2(ratio)) + 1):
        I1LRU = np.zeros((b, 2 ** z * r, 2 ** z * c))
        if first:
            I1LRU[:, 1:I1LRU.shape[1]:2, 1:I1LRU.shape[2]:2] = image
            first = 0
        else:
            I1LRU[:, 0:I1LRU.shape[1]:2, 0:I1LRU.shape[2]:2] = image

        for ii in range(0, b):
            t = I1LRU[ii, :, :]
            for j in range(0, t.shape[0]):
                t[j, :] = ndimage.correlate(t[j, :], BaseCoeff, mode='wrap')
            for k in range(0, t.shape[1]):
                t[:, k] = ndimage.correlate(t[:, k], BaseCoeff, mode='wrap')
            I1LRU[ii, :, :] = t
        image = I1LRU

    re_image = np.transpose(I1LRU, (1, 2, 0))

    return re_image
    
def kaiser2d(N, beta):
    
    t=np.arange(-(N-1)/2,(N+1)/2)/np.double(N-1)
    t1,t2=np.meshgrid(t,t)
    t12=np.sqrt(t1*t1+t2*t2)
    w1=np.kaiser(N,beta)
    w=np.interp(t12,t,w1)
    w[t12>t[-1]]=0
    w[t12<t[0]]=0
    
    return w

def fir_filter_wind(Hd,w):
	
    hd=np.rot90(np.fft.fftshift(np.rot90(Hd,2)),2)
    h=np.fft.fftshift(np.fft.ifft2(hd))
    h=np.rot90(h,2)
    h=h*w
    h=h/np.sum(h)
    
    return h

def MTF_GLP_HPM_torch(pan, u_hs, sensor='gaussian'):

    batch, C, X, Y = u_hs.shape   # tensor: batch, C, M, N

    ratio = 4

    ## equalization
    image_hr = pan.repeat(1,C,1,1)  # tensor: [batch, C, M, N]
    image_hr = (image_hr - torch.mean(image_hr,dim=(2,3), keepdim=True)) \
               * (torch.std(u_hs, dim=(2,3),unbiased=True,keepdim=True)
                  / torch.std(image_hr, dim=(2,3),unbiased=True,keepdim=True)) + torch.mean(u_hs, dim=(2,3), keepdim=True)  # tensor: [batch, C, M, N]
    pan_lp = torch.zeros_like(u_hs)

    ## MTF
    N =41
    fcut = 1/ratio
    match = 0

    if sensor == 'gaussian':
        sig = (1/(2*(2.772587)/ratio**2))**0.5
        kernel = np.multiply(cv2.getGaussianKernel(9, sig), cv2.getGaussianKernel(9,sig).T)
        kernel = np.expand_dims(np.expand_dims(kernel, 0), 0).astype(np.float32)
        kernel = torch.from_numpy(kernel).to(pan.device, dtype=pan.dtype)  # tensor: [1,1,9,9]
        kernel = kernel.repeat(1,4,1,1)  # tensor: [1,4,9,9]
        
        temp = F.conv2d(image_hr, kernel, stride=1, padding=4)  # tensor: [batch, C, M, N]
        # temp = temp[:, :, 0::ratio, 0::ratio]  # batch, C, M//4, N//4
        temp = downsample_nearest(temp, ratio)
        pan_lp = upsample_bicubic(temp, ratio)   # tensor: [batch, C, M, N]
    
    elif sensor == None:
        match=1
        GNyq = 0.3*np.ones((C,))
    elif sensor=='QB':
        match=1
        GNyq = np.asarray([0.34, 0.32, 0.30, 0.22],dtype='float32')    # Band Order: B,G,R,NIR
    elif sensor=='IKONOS':
        match=1           #MTF usage
        GNyq = np.asarray([0.26,0.28,0.29,0.28],dtype='float32')    # Band Order: B,G,R,NIR
    elif sensor=='GeoEye1':
        match=1             # MTF usage
        GNyq = np.asarray([0.23,0.23,0.23,0.23],dtype='float32')    # Band Order: B,G,R,NIR   
    elif sensor=='WV2':
        match=1            # MTF usage
        GNyq = [0.35,0.35,0.35,0.35,0.35,0.35,0.35,0.27]
    elif sensor=='WV3':
        match=1             #MTF usage
        GNyq = 0.29 * np.ones(8)
    else:
        match = 1
        GNyq = np.asarray([0.3, 0.3, 0.3, 0.3])

    if match==1:
        t = []
        for i in range(C):
            alpha = np.sqrt((N * (fcut / 2))** 2 / (-2 * np.log(GNyq)))
            H = np.multiply(cv2.getGaussianKernel(N, alpha[i]), cv2.getGaussianKernel(N, alpha[i]).T)
            HD = H / np.max(H)

            h = fir_filter_wind(HD, kaiser2d(N, 0.5))  # [41,41]

            h = np.real(h).astype(np.float32)
            h = np.expand_dims(np.expand_dims(h, 0),0)
            h = torch.from_numpy(h).to(pan.device, dtype=pan.dtype)  # tensor: [1,1,N,N]

            pad = (20,20,20,20)
            temp = torch.nn.functional.pad(image_hr[:, i, :, :].unsqueeze(1), pad, mode='replicate')
            temp = F.conv2d(temp, h, stride=1, padding=0)
            temp = downsample_nearest(temp, ratio)   # tensor: [batch, 1, X//4, Y//4]

            t.append(temp)

        if C ==4:
            t = torch.cat((t[0],t[1],t[2],t[3]),1)    # tensor: [batch, 4, X//4, Y//4]
        else:
            t = torch.cat((t[0],t[1],t[2],t[3],t[4],t[5],t[6],t[7]),1)    # tensor: [batch, 4, X//4, Y//4]

        pan_lp = upsample_bicubic(t, ratio)  # tensor: [batch, 4, X, Y]

    I_MTF_GLP_HPM = u_hs*(image_hr/(pan_lp+1e-8))

    return I_MTF_GLP_HPM


class senet(nn.Module):
    def __init__(self, spectral_num=4):
        super(senet, self).__init__()

        self.module1 = nn.Sequential(
            nn.Conv2d(in_channels=spectral_num, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        self.module2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=spectral_num, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(32, 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = x
        x = self.module1(x)
        se = x
        x = self.se(x)
        x = torch.mul(x, se)
        x = self.module2(x)
        x =x * x1
        return x
    

class MTF_GLP_HPM(nn.Module):
    def __init__(self, spectral_num = 4):
        super(MTF_GLP_HPM, self).__init__()

        self.conv2 = senet(spectral_num = spectral_num)

    def forward(self, lrms, pan, sensor):
        fea = MTF_GLP_HPM_torch(pan,lrms,sensor)
        out = self.conv2(fea)
        return out
    
