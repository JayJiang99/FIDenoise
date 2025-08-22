import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import itertools
from model.warplayer import warp
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class RepConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=None, groups=1,
                 map_k=3):
        super(RepConv, self).__init__()
        assert map_k <= kernel_size
        self.origin_kernel_shape = (out_channels, in_channels // groups, kernel_size, kernel_size)
        self.register_buffer('weight', torch.zeros(*self.origin_kernel_shape))
        G = in_channels * out_channels // (groups ** 2)
        self.num_2d_kernels = out_channels * in_channels // groups
        self.kernel_size = kernel_size
        self.convmap = nn.Conv2d(in_channels=self.num_2d_kernels,
                                 out_channels=self.num_2d_kernels, kernel_size=map_k, stride=1, padding=map_k // 2,
                                 groups=G, bias=False)
        #nn.init.zeros_(self.convmap.weight)
        self.bias = None#nn.Parameter(torch.zeros(out_channels), requires_grad=True)     # must have a bias for identical initialization
        self.stride = stride
        self.groups = groups
        if padding is None:
            padding = kernel_size // 2
        self.padding = padding

    def forward(self, inputs):
        origin_weight = self.weight.view(1, self.num_2d_kernels, self.kernel_size, self.kernel_size)
        kernel = self.weight + self.convmap(origin_weight).view(*self.origin_kernel_shape)
        return F.conv2d(inputs, kernel, stride=self.stride, padding=self.padding, dilation=1, groups=self.groups, bias=self.bias)

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        RepConv(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding),
        nn.PReLU(out_planes)
        )

def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        torch.nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes, kernel_size=4, stride=2, padding=1, bias=True),
        nn.PReLU(out_planes)
        )
            
class Conv2(nn.Module):
    def __init__(self, in_planes, out_planes, stride=2):
        super(Conv2, self).__init__()
        self.conv1 = conv(in_planes, out_planes, 3, stride, 1)
        self.conv2 = conv(out_planes, out_planes, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
    
c = 16
class Contextnet(nn.Module):
    def __init__(self):
        super(Contextnet, self).__init__()
        self.conv1 = Conv2(3, c)
        self.conv2 = Conv2(c, 2*c)
        self.conv3 = Conv2(2*c, 4*c)
        self.conv4 = Conv2(4*c, 8*c)
    
    def forward(self, x, flow):
        x = self.conv1(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False, recompute_scale_factor=False) * 0.5
        f1 = warp(x, flow)        
        x = self.conv2(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False, recompute_scale_factor=False) * 0.5
        f2 = warp(x, flow)
        x = self.conv3(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False, recompute_scale_factor=False) * 0.5
        f3 = warp(x, flow)
        x = self.conv4(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False, recompute_scale_factor=False) * 0.5
        f4 = warp(x, flow)
        return [f1, f2, f3, f4]
    
class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.down0 = Conv2(17, 2*c)
        self.down1 = Conv2(4*c, 4*c)
        self.down2 = Conv2(8*c, 8*c)
        self.down3 = Conv2(16*c, 16*c)
        self.up0 = deconv(32*c, 8*c)
        self.up1 = deconv(16*c, 4*c)
        self.up2 = deconv(8*c, 2*c)
        self.up3 = deconv(4*c, c)
        self.conv = nn.Conv2d(c, 3, 3, 1, 1)
        self.pixel_shuffle = CustomPixelShuffle(3)

    def forward(self, img0, img1, warped_img0, warped_img1, mask, flow, c0, c1):
        s0 = self.down0(torch.cat((img0, img1, warped_img0, warped_img1, mask, flow), 1))
        s1 = self.down1(torch.cat((s0, c0[0], c1[0]), 1))
        s2 = self.down2(torch.cat((s1, c0[1], c1[1]), 1))
        s3 = self.down3(torch.cat((s2, c0[2], c1[2]), 1))
        x = self.up0(torch.cat((s3, c0[3], c1[3]), 1))
        x = self.up1(torch.cat((x, s2), 1)) 
        x = self.up2(torch.cat((x, s1), 1)) 
        x = self.up3(torch.cat((x, s0), 1)) 
        x = self.conv(x)
        # x = self.pixel_shuffle(x)
        return torch.sigmoid(x)

class ModifiedUnet(nn.Module):
    def __init__(self):
        super(ModifiedUnet, self).__init__()
        self.down0 = Conv2(14, 2*c)
        self.down1 = Conv2(2*c, 4*c)
        self.down2 = Conv2(4*c, 8*c)
        self.down3 = Conv2(8*c, 16*c)
        self.up0 = deconv(16*c, 8*c)
        self.up1 = deconv(16*c, 4*c)
        self.up2 = deconv(8*c, 2*c)
        self.up3 = deconv(4*c, c)
        self.conv = nn.Conv2d(c, 3, 3, 1, 1)
        self.pixel_shuffle = CustomPixelShuffle(3)

    def forward(self, x):
        # Encoder
        s0 = self.down0(x)
        s1 = self.down1(s0)
        s2 = self.down2(s1)
        s3 = self.down3(s2)

        # Decoder
        x = self.up0(s3)
        x = self.up1(torch.cat((x, s2), 1))
        x = self.up2(torch.cat((x, s1), 1))
        x = self.up3(torch.cat((x, s0), 1))
        x = self.conv(x)
        
        # Custom pixel shuffle to achieve Bx1xHx3W
        x = self.pixel_shuffle(x)
        
        return torch.sigmoid(x)

class Conv2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.PReLU()
        )
        self.downsample = nn.MaxPool2d(2, 2)

    def forward(self, x):
        return self.downsample(self.conv(x))

def deconv(in_planes, out_planes):
    return nn.Sequential(
        torch.nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes, kernel_size=4, stride=2, padding=1),
        nn.PReLU(out_planes)
    )

class CustomPixelShuffle(nn.Module):
    def __init__(self, upscale_factor):
        super(CustomPixelShuffle, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        B, C, H, W = x.size()
        x = x.view(B, 1, C, H, W)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(B, 1, H, W * self.upscale_factor)
        return x