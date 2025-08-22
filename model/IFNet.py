import torch
import torch.nn as nn
import torch.nn.functional as F
from model.warplayer import warp
from model.refine import *
from model.arch_unet import *



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


def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        torch.nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes, kernel_size=4, stride=2, padding=1),
        nn.PReLU(out_planes)
    )

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
    )

def repconv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        RepConv(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding),
        nn.PReLU(out_planes)
    )


# ----------------------
# Learned upsampling modules
# ----------------------
class LearnedWidthUpsampler(nn.Module):
    def __init__(self, in_channels: int = 1, hidden_channels: int = 32, process_as_gray: bool = True):
        super().__init__()
        self.process_as_gray = process_as_gray
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1, bias=True),
            nn.PReLU(hidden_channels),
            nn.Conv2d(hidden_channels, in_channels, 3, padding=1, bias=True),
        )
        self.alpha = nn.Parameter(torch.tensor(0.5))

        
    def forward(self, x: torch.Tensor, out_size):
        target_h, target_w = out_size
        if self.process_as_gray and x.shape[1] == 3:
            x_gray = x.mean(dim=1, keepdim=True)
            y_bilinear_gray = F.interpolate(x_gray, size=(target_h, target_w), mode='bilinear', align_corners=False)
            y_refined_gray = self.refine(y_bilinear_gray)
            gate = torch.sigmoid(self.alpha)
            y_gray = y_refined_gray * (1.0 - gate) + y_bilinear_gray * gate
            return y_gray.repeat(1, 3, 1, 1)
        else:
            y_bilinear = F.interpolate(x, size=(target_h, target_w), mode='bilinear', align_corners=False)
            y_refined = self.refine(y_bilinear)
            gate = torch.sigmoid(self.alpha)
            return y_refined * (1.0 - gate) + y_bilinear * gate


def convex_upsample_aniso(x: torch.Tensor, mask: torch.Tensor, up_h: int = 1, up_w: int = 3) -> torch.Tensor:
    """
    RAFT-style convex upsampling generalized to anisotropic up factors.
    x:    [B, C, H, W]
    mask: [B, 9*up_h*up_w, H, W]
    returns: [B, C, H*up_h, W*up_w]
    """
    B, C, H, W = x.shape
    # unfold 3x3 neighborhoods: [B, C*9, H*W] -> [B, C, 9, H, W]
    neigh = F.unfold(x, kernel_size=3, padding=1).view(B, C, 9, H, W)

    # reshape mask to [B, 1, 9, up_h, up_w, H, W] and softmax over the 3x3 neighbors (dim=2)
    mask = mask.view(B, 1, 9, up_h, up_w, H, W)
    mask = torch.softmax(mask, dim=2)

    # weighted sum over the 3x3 neighbors -> [B, C, up_h, up_w, H, W]
    up_x = (neigh.unsqueeze(3).unsqueeze(3) * mask).sum(dim=2)

    # reassemble subpixels to full-res: [B, C, H, up_h, W, up_w] -> [B, C, H*up_h, W*up_w]
    up_x = up_x.permute(0, 1, 4, 2, 5, 3).contiguous().view(B, C, H * up_h, W * up_w)
    return up_x


class DynamicKernelUpsamplerAniso(nn.Module):
    """
    CARAFE-like content-aware upsampling with anisotropic factors.
    Predicts per-location kernels and reassembles HR features.
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 1, up_h: int = 1, up_w: int = 3, k: int = 5, hidden: int = 64, process_as_gray: bool = True):
        super().__init__()
        assert k >= 1 and (k % 2 in (0, 1))
        self.up_h, self.up_w, self.k = up_h, up_w, k
        self.pad = k // 2
        self.process_as_gray = process_as_gray

        # Kernel prediction branch (predicts k*k*up_h*up_w weights per location)
        self.kernel_pred = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, (k * k) * (up_h * up_w), 1)
        )
        # Optional channel mixer after reassembly
        self.post = nn.Conv2d(in_channels, out_channels, 1)

        # Initialize last layer to small so start ~bilinear-ish after softmax
        nn.init.zeros_(self.kernel_pred[-1].weight)
        nn.init.zeros_(self.kernel_pred[-1].bias)

    def forward(self, x: torch.Tensor, out_size):
        target_h, target_w = out_size
        B, C, H, W = x.shape
        # Validate up factors match target size; fallback to bilinear otherwise
        if (target_h % H != 0) or (target_w % W != 0):
            y = F.interpolate(x, size=(target_h, target_w), mode='bilinear', align_corners=False)
            return y
        up_h = target_h // H
        up_w = target_w // W
        if up_h != self.up_h or up_w != self.up_w:
            y = F.interpolate(x, size=(target_h, target_w), mode='bilinear', align_corners=False)
            return y

        if self.process_as_gray and C == 3:
            x = x.mean(dim=1, keepdim=True)
            C = 1

        if C != 1:
            # Only 1-channel supported when not processing as gray; fallback
            y = F.interpolate(x, size=(target_h, target_w), mode='bilinear', align_corners=False)
            return y

        k = self.k
        # Predict kernels (logits): [B, k*k*up_h*up_w, H, W] -> [B, k2, up_h, up_w, H, W]
        ker = self.kernel_pred(x).view(B, (k * k), self.up_h, self.up_w, H, W)
        ker = torch.softmax(ker, dim=1)

        # Extract kxk neighborhoods: [B, C*k*k, H*W] -> [B, C, k2, H, W]
        neigh = F.unfold(x, kernel_size=k, padding=self.pad).view(B, C, k * k, H, W)

        # Weighted sum per subpixel -> [B, C, up_h, up_w, H, W]
        out = (neigh.unsqueeze(3).unsqueeze(3) * ker.unsqueeze(1)).sum(dim=2)

        # Rearrange subpixels to HR -> [B, C, H*up_h, W*up_w]
        out = out.permute(0, 1, 4, 2, 5, 3).contiguous().view(B, C, H * self.up_h, W * self.up_w)

        out = self.post(out)  # keep channels=1
        # replicate to 3 channels if needed to match interface
        if out.shape[1] == 1:
            out = out.repeat(1, 3, 1, 1)
        return out


class ConvexWidthUpsampler(nn.Module):
    def __init__(self, in_channels: int = 1, hidden_channels: int = 32, process_as_gray: bool = True, up_h: int = 1, up_w: int = 3):
        super().__init__()
        self.process_as_gray = process_as_gray
        self.up_h = up_h
        self.up_w = up_w
        # Mask prediction at low resolution: outputs 9 * up_h * up_w channels
        self.mask_net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1, bias=True),
            nn.PReLU(hidden_channels),
            nn.Conv2d(hidden_channels, 9 * up_h * up_w, 1, bias=True),
        )

    def forward(self, x: torch.Tensor, out_size):
        target_h, target_w = out_size
        B, C, H, W = x.shape
        # Validate up factors match target size; otherwise fallback
        if (target_h % H != 0) or (target_w % W != 0):
            return F.interpolate(x, size=(target_h, target_w), mode='bilinear', align_corners=False)
        up_h = target_h // H
        up_w = target_w // W
        if up_h != self.up_h or up_w != self.up_w:
            return F.interpolate(x, size=(target_h, target_w), mode='bilinear', align_corners=False)

        if self.process_as_gray and x.shape[1] == 3:
            x_gray = x.mean(dim=1, keepdim=True)
            mask = self.mask_net(x_gray)
            y_gray = convex_upsample_aniso(x_gray, mask, up_h=self.up_h, up_w=self.up_w)
            return y_gray.repeat(1, 3, 1, 1)
        else:
            # If not grayscale processing, expect channels==1; otherwise fallback to bilinear
            if x.shape[1] != 1:
                return F.interpolate(x, size=(target_h, target_w), mode='bilinear', align_corners=False)
            mask = self.mask_net(x)
            return convex_upsample_aniso(x, mask, up_h=self.up_h, up_w=self.up_w)


class GroupBatchnorm2d(nn.Module):
    def __init__(self, c_num:int, 
                 group_num:int = 16, 
                 eps:float = 1e-10
                 ):
        super(GroupBatchnorm2d,self).__init__()
        assert c_num    >= group_num
        self.group_num  = group_num
        self.weight     = nn.Parameter( torch.randn(c_num, 1, 1)    )
        self.bias       = nn.Parameter( torch.zeros(c_num, 1, 1)    )
        self.eps        = eps
    def forward(self, x):
        N, C, H, W  = x.size()
        x           = x.view(   N, self.group_num, -1   )
        mean        = x.mean(   dim = 2, keepdim = True )
        std         = x.std (   dim = 2, keepdim = True )
        x           = (x - mean) / (std+self.eps)
        x           = x.view(N, C, H, W)
        return x * self.weight + self.bias


class SRU(nn.Module):
    def __init__(self,
                 oup_channels:int, 
                 group_num:int = 16,
                 gate_treshold:float = 0.5,
                 torch_gn:bool = True
                 ):
        super().__init__()
        
        self.gn             = nn.GroupNorm( num_channels = oup_channels, num_groups = group_num ) if torch_gn else GroupBatchnorm2d(c_num = oup_channels, group_num = group_num)
        self.gate_treshold  = gate_treshold
        self.sigomid        = nn.Sigmoid()

    def forward(self,x):
        gn_x        = self.gn(x)
        w_gamma     = self.gn.weight/sum(self.gn.weight)
        w_gamma     = w_gamma.view(1,-1,1,1)
        reweigts    = self.sigomid( gn_x * w_gamma )
        # Gate
        w1          = torch.where(reweigts > self.gate_treshold, torch.ones_like(reweigts), reweigts) # 大于门限值的设为1，否则保留原值
        w2          = torch.where(reweigts > self.gate_treshold, torch.zeros_like(reweigts), reweigts) # 大于门限值的设为0，否则保留原值
        x_1         = w1 * x
        x_2         = w2 * x
        y           = self.reconstruct(x_1,x_2)
        return y
    
    def reconstruct(self,x_1,x_2):
        x_11,x_12 = torch.split(x_1, x_1.size(1)//2, dim=1)
        x_21,x_22 = torch.split(x_2, x_2.size(1)//2, dim=1)
        return torch.cat([ x_11+x_22, x_12+x_21 ],dim=1)


class CRU(nn.Module):
    '''
    alpha: 0<alpha<1
    '''
    def __init__(self, 
                 op_channel:int,
                 alpha:float = 1/2,
                 squeeze_radio:int = 2 ,
                 group_size:int = 2,
                 group_kernel_size:int = 3,
                 ):
        super().__init__()
        self.up_channel     = up_channel   =   int(alpha*op_channel)
        self.low_channel    = low_channel  =   op_channel-up_channel
        self.squeeze1       = nn.Conv2d(up_channel,up_channel//squeeze_radio,kernel_size=1,bias=False)
        self.squeeze2       = nn.Conv2d(low_channel,low_channel//squeeze_radio,kernel_size=1,bias=False)
        #up
        self.GWC            = nn.Conv2d(up_channel//squeeze_radio, op_channel,kernel_size=group_kernel_size, stride=1,padding=group_kernel_size//2, groups = group_size)
        self.PWC1           = nn.Conv2d(up_channel//squeeze_radio, op_channel,kernel_size=1, bias=False)
        #low
        self.PWC2           = nn.Conv2d(low_channel//squeeze_radio, op_channel-low_channel//squeeze_radio,kernel_size=1, bias=False)
        self.advavg         = nn.AdaptiveAvgPool2d(1)

    def forward(self,x):
        # Split
        up,low  = torch.split(x,[self.up_channel,self.low_channel],dim=1)
        up,low  = self.squeeze1(up),self.squeeze2(low)
        # Transform
        Y1      = self.GWC(up) + self.PWC1(up)
        Y2      = torch.cat( [self.PWC2(low), low], dim= 1 )
        # Fuse
        out     = torch.cat( [Y1,Y2], dim= 1 )
        out     = F.softmax( self.advavg(out), dim=1 ) * out
        out1,out2 = torch.split(out,out.size(1)//2,dim=1)
        return out1+out2


class ScConv(nn.Module):
    def __init__(self,
                op_channel:int,
                group_num:int = 4,
                gate_treshold:float = 0.5,
                alpha:float = 1/2,
                squeeze_radio:int = 2 ,
                group_size:int = 2,
                group_kernel_size:int = 3,
                 ):
        super().__init__()
        self.SRU = SRU( op_channel, 
                       group_num            = group_num,  
                       gate_treshold        = gate_treshold )
        self.CRU = CRU( op_channel, 
                       alpha                = alpha, 
                       squeeze_radio        = squeeze_radio ,
                       group_size           = group_size ,
                       group_kernel_size    = group_kernel_size )
    
    def forward(self,x):
        x = self.SRU(x)
        x = self.CRU(x)
        return x
class IFBlock(nn.Module):
    def __init__(self, in_planes, c=64):
        super(IFBlock, self).__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c//2, 3, 2, 1),
            conv(c//2, c, 3, 2, 1),
            )
        self.convblock = nn.Sequential(
            ScConv(c, c),
            ScConv(c, c),
            ScConv(c, c),
            ScConv(c, c),

        )
        self.lastconv = nn.ConvTranspose2d(c, 5, 4, 2, 1)

    def forward(self, x, flow, scale):
        if scale != 1:
            x = F.interpolate(x, scale_factor = 1. / scale, mode="bilinear", align_corners=False)
        if flow != None:
            flow = F.interpolate(flow, scale_factor = 1. / scale, mode="bilinear", align_corners=False) * 1. / scale
            x = torch.cat((x, flow), 1)
        x = self.conv0(x)
        x = self.convblock(x) + x
        tmp = self.lastconv(x)
        tmp = F.interpolate(tmp, scale_factor = scale * 2, mode="bilinear", align_corners=False)
        flow = tmp[:, :4] * scale * 2
        mask = tmp[:, 4:5]
        return flow, mask    
class IFNet(nn.Module):
    def __init__(self, upsample_mode: str = 'learned', grayscale_triplet: bool = True):
        super(IFNet, self).__init__()
        self.block0 = IFBlock(6, c=240)
        self.block1 = IFBlock(13+4, c=160)
        self.block2 = IFBlock(13+4, c=80)
        self.block_tea = IFBlock(16+4, c=80)
        self.contextnet = Contextnet()
        self.unet = Unet()
        # self.unetUpsample = ModifiedUnet()
        # Upsampling modules
        self.upsample_mode = upsample_mode
        self.grayscale_triplet = grayscale_triplet
        self.width_upsampler_learned = LearnedWidthUpsampler(in_channels=1, hidden_channels=32, process_as_gray=self.grayscale_triplet)
        self.width_upsampler_convex = ConvexWidthUpsampler(in_channels=1, hidden_channels=32, process_as_gray=self.grayscale_triplet)
        self.width_upsampler_carafe = DynamicKernelUpsamplerAniso(in_channels=1, out_channels=1, up_h=1, up_w=3, k=5, hidden=32, process_as_gray=self.grayscale_triplet)
 
    def forward(self, x, scale=[4,2,1], timestep=0.5):
        img0 = x[:, :3]
        img1 = x[:, 3:6]
        gt = x[:, 6:] # In inference time, gt is None
        flow_list = []
        merged = []
        mask_list = []
        warped_img0 = img0
        warped_img1 = img1
        flow = None 
        loss_distill = 0
        stu = [self.block0, self.block1, self.block2]
        for i in range(3):
            if flow != None:
                flow_d, mask_d = stu[i](torch.cat((img0, img1, warped_img0, warped_img1, mask), 1), flow, scale=scale[i])
                flow = flow + flow_d
                mask = mask + mask_d
            else:
                flow, mask = stu[i](torch.cat((img0, img1), 1), None, scale=scale[i])
            mask_list.append(torch.sigmoid(mask))
            flow_list.append(flow)
            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])
            merged_student = (warped_img0, warped_img1)
            merged.append(merged_student)
        if gt.shape[1] == 3:
            flow_d, mask_d = self.block_tea(torch.cat((img0, img1, warped_img0, warped_img1, mask, gt), 1), flow, scale=1)
            flow_teacher = flow + flow_d
            warped_img0_teacher = warp(img0, flow_teacher[:, :2])
            warped_img1_teacher = warp(img1, flow_teacher[:, 2:4])
            mask_teacher = torch.sigmoid(mask + mask_d)
            merged_teacher = warped_img0_teacher * mask_teacher + warped_img1_teacher * (1 - mask_teacher)
        else:
            flow_teacher = None
            merged_teacher = None
        for i in range(3):
            merged[i] = merged[i][0] * mask_list[i] + merged[i][1] * (1 - mask_list[i])
            if gt.shape[1] == 3:
                loss_mask = ((merged[i] - gt).abs().mean(1, True) > (merged_teacher - gt).abs().mean(1, True) + 0.01).float().detach()
                loss_distill += (((flow_teacher.detach() - flow_list[i]) ** 2).mean(1, True) ** 0.5 * loss_mask).mean()
        c0 = self.contextnet(img0, flow[:, :2])
        c1 = self.contextnet(img1, flow[:, 2:4])
        tmp = self.unet(img0, img1, warped_img0, warped_img1, mask, flow, c0, c1)
        res = tmp[:, :3] * 2 - 1
        merged[2] = torch.clamp(merged[2] + res, 0, 1)
        
        # obtain target size by multiplying the width of merged[2] by 3 and keep the height the same
        target_size = (merged[2].shape[2], merged[2].shape[3]*3)
        # print("target_size", target_size)
        if self.upsample_mode == 'learned':
            upsampled_merged = self.width_upsampler_learned(merged[2], target_size)
        elif self.upsample_mode == 'convex':
            upsampled_merged = self.width_upsampler_convex(merged[2], target_size)
        elif self.upsample_mode == 'carafe':
            upsampled_merged = self.width_upsampler_carafe(merged[2], target_size)
        else:
            upsampled_merged = F.interpolate(merged[2], size=target_size, mode='bilinear', align_corners=False)
        
        # print the shape of the upsampled_merged   
        # print("upsampled_merged.shape", upsampled_merged.shape)
         
         
        return flow_list, mask_list[2], merged, flow_teacher, merged_teacher, loss_distill, upsampled_merged
