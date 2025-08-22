import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
import torch.optim as optim
import itertools
from model.warplayer import warp
from torch.nn.parallel import DistributedDataParallel as DDP
from model.IFNet import *
from model.IFNet_m import *
import torch.nn.functional as F
from model.loss import *
from model.laplacian import *
from model.refine import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
operation_seed_counter = 0

class Model:
    def __init__(self, local_rank=-1, arbitrary=False):
        if arbitrary == True:
            self.flownet = IFNet_m()
        else:
            self.flownet = IFNet(upsample_mode='convex')
        self.device()
        self.optimG = AdamW(self.flownet.parameters(), lr=1e-6, weight_decay=1e-3) # use large weight decay may avoid NaN loss
        self.epe = EPE()
        self.lap = LapLoss()
        self.sobel = SOBEL()
        if local_rank != -1:
            self.flownet = DDP(self.flownet, device_ids=[local_rank], output_device=local_rank)

    def train(self):
        self.flownet.train()

    def eval(self):
        self.flownet.eval()

    def device(self):
        self.flownet.to(device)

    def load_model(self, path, rank=0):
        def convert(param):
            return {
            k.replace("module.", ""): v
                for k, v in param.items()
                if "module." in k
            }
            
        if rank <= 0:
            # self.flownet.load_state_dict(convert(torch.load('{}/flownet.pkl'.format(path))))
            # load pth file
            self.flownet = torch.load(path)
            
        
    def save_model(self, path, rank=0):
        if rank == 0:
            torch.save(self.flownet.state_dict(),'{}/fidenoise_model.pkl'.format(path))
        # I want to save both weight and model structure
        torch.save(self.flownet, '{}/fidenoise_model.pth'.format(path))

    def inference(self, img0, img1, scale=1, scale_list=[4, 2, 1], TTA=False, timestep=0.5):
        for i in range(3):
            scale_list[i] = scale_list[i] * 1.0 / scale
        imgs = torch.cat((img0, img1), 1)
        flow, mask, merged, flow_teacher, merged_teacher, loss_distill = self.flownet(imgs, scale_list, timestep=timestep)
        if TTA == False:
            return merged[2]
        else:
            flow2, mask2, merged2, flow_teacher2, merged_teacher2, loss_distill2 = self.flownet(imgs.flip(2).flip(3), scale_list, timestep=timestep)
            return (merged[2] + merged2[2].flip(2).flip(3)) / 2
    def inference_denoise(self, img0, img1, gt, scale=1, scale_list=[4, 2, 1], TTA=False, timestep=0.5):
        for i in range(3):
            scale_list[i] = scale_list[i] * 1.0 / scale
        imgs = torch.cat((img0, img1, gt), 1)
        flow, mask, merged, flow_teacher, merged_teacher, loss_distill, upsampled_merged = self.flownet(imgs, scale_list, timestep=timestep)

        return merged[2], merged_teacher[0], upsampled_merged, flow
    
    def interleave_tensors(self, t1, t2, t3):
        # Assuming t1, t2, t3 are of shape [B, C, H, W]
        B, C, H, W = t1.shape
        result = torch.zeros(B, C, H, W*3, device=t1.device)
        result[:, :, :, 0::3] = t1
        result[:, :, :, 1::3] = t2
        result[:, :, :, 2::3] = t3
        return result
    
    
    def get_generator(self):
        global operation_seed_counter
        operation_seed_counter += 1
        g_cuda_generator = torch.Generator(device=device)
        g_cuda_generator.manual_seed(operation_seed_counter)
        return g_cuda_generator


    def space_to_depth(self, x, block_size):
        n, c, h, w = x.size()
        unfolded_x = torch.nn.functional.unfold(x, block_size, stride=block_size)
        return unfolded_x.view(n, c * block_size**2, h // block_size,
                            w // block_size)
    def generate_mask_pair(self, img):
        # prepare masks (N x C x H/2 x W/2)
        n, c, h, w = img.shape
        mask1 = torch.zeros(size=(n * h // 2 * w // 2 * 4, ),
                        dtype=torch.bool,
                        device=img.device)
        mask2 = torch.zeros(size=(n * h // 2 * w // 2 * 4, ),
                        dtype=torch.bool,
                        device=img.device)
        # prepare random mask pairs
        idx_pair = torch.tensor(
            [[0, 1], [0, 2], [1, 3], [2, 3], [1, 0], [2, 0], [3, 1], [3, 2]],
        dtype=torch.int64,
        device=img.device)
        rd_idx = torch.zeros(size=(n * h // 2 * w // 2, ),
                            dtype=torch.int64,
                            device=img.device)
        torch.randint(low=0,
                    high=8,
                    size=(n * h // 2 * w // 2, ),
                    generator=self.get_generator(),
                    out=rd_idx)
        rd_pair_idx = idx_pair[rd_idx]
        rd_pair_idx += torch.arange(start=0,
                                end=n * h // 2 * w // 2 * 4,
                                step=4,
                                dtype=torch.int64,
                                device=img.device).reshape(-1, 1)
        # get masks
        mask1[rd_pair_idx[:, 0]] = 1
        mask2[rd_pair_idx[:, 1]] = 1
        return mask1, mask2


    def generate_subimages(self, img, mask):
        n, c, h, w = img.shape
        subimage = torch.zeros(n,
                           c,
                           h // 2,
                           w // 2,
                           dtype=img.dtype,
                           layout=img.layout,
                           device=img.device)
        # per channel
        for i in range(c):
            img_per_channel = self.space_to_depth(img[:, i:i + 1, :, :], block_size=2)
        img_per_channel = img_per_channel.permute(0, 2, 3, 1).reshape(-1)
        subimage[:, i:i + 1, :, :] = img_per_channel[mask].reshape(
                n, h // 2, w // 2, 1).permute(0, 3, 1, 2)
        return subimage
    
    def compute_image_gradients(self, image):
        # Calculate horizontal gradient (difference between adjacent columns)
        grad_x = image[:, :, :, :-1] - image[:, :, :, 1:]
        # Calculate vertical gradient (difference between adjacent rows)
        grad_y = image[:, :, :-1, :] - image[:, :, 1:, :]
        # add a column of zeros to the end of the grad_x 
        grad_x = torch.cat((grad_x, torch.zeros_like(grad_x[:, :, :, :1])), dim=3)
        # add a row of zeros to the end of the grad_y
        grad_y = torch.cat((grad_y, torch.zeros_like(grad_y[:, :, :1, :])), dim=2)

        return grad_x, grad_y


    def compute_gradient_loss(self, pred,gt):
        binary_mask = (gt > (gt.max()/4)).float()
        grad_interleaved_mergedx, grad_interleaved_mergedy   = self.compute_image_gradients(gt)
        grad_upsampled_mergedx, grad_upsampled_mergedy = self.compute_image_gradients(pred)
        grad_interleaved_mergedx = grad_interleaved_mergedx * binary_mask
        grad_upsampled_mergedx = grad_upsampled_mergedx * binary_mask
        grad_interleaved_mergedy = grad_interleaved_mergedy * binary_mask
        grad_upsampled_mergedy = grad_upsampled_mergedy * binary_mask
        return (F.mse_loss(grad_interleaved_mergedx , grad_upsampled_mergedx ) + F.mse_loss(grad_interleaved_mergedy , grad_upsampled_mergedy)).mean()
    def update(self, imgs, gt, original_gt, learning_rate=None, mul=1, training=True, flow_gt=None, loss_weights=None):
        if learning_rate is not None:
            for param_group in self.optimG.param_groups:
                param_group['lr'] = learning_rate
        img0 = imgs[:, :3]
        img1 = imgs[:, 3:]
        if training:
            self.train()
        else:
            self.eval()
        flow, mask, merged, flow_teacher, merged_teacher, loss_distill, upsampled_merged = self.flownet(torch.cat((imgs, gt), 1), scale=[4, 2, 1])

        loss_l1 = (self.lap(merged[2], gt)).mean()
        loss_l1_original = (self.lap(upsampled_merged, original_gt)).mean()
        loss_tea = 0.0*(self.lap(merged_teacher, gt)).mean()
        extracted_merged = upsampled_merged[:, :, :, 1::3]
        loss_l1_extracted = (self.lap(extracted_merged, gt)).mean()

        loss_l1_mask = self.compute_gradient_loss(merged[2], gt)
        loss_l1_mask_original = self.compute_gradient_loss(upsampled_merged, original_gt)

        # Default weights if none passed
        if loss_weights is None:
            loss_weights = {
                'w_main': 1.0,
                'w_up': 0.1,
                'w_grad': 0.5,
                'w_grad_orig': 0.1,
                'w_tea': 1.0,
                'w_distill': 0.002,
            }
        w_main = loss_weights.get('w_main', 1.0)
        w_up = loss_weights.get('w_up', 0.1)
        w_grad = loss_weights.get('w_grad', 0.5)
        w_grad_orig = loss_weights.get('w_grad_orig', 0.1)
        w_tea = loss_weights.get('w_tea', 1.0)
        

        # Compute total loss (for logging consistency in eval as well)
        loss_total = (
            w_grad * loss_l1_mask +
            w_grad_orig * loss_l1_mask_original +
            w_main * loss_l1 +
            w_up * loss_l1_original 
        )

        if training:
            self.optimG.zero_grad()
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(self.flownet.parameters(), max_norm=1.0)
            self.optimG.step()
        else:
            flow_teacher = flow[2]
        return upsampled_merged, {
            'merged_tea': merged_teacher,
            'upsampled_merged': upsampled_merged,
            'mask': mask,
            'mask_tea': mask,
            'flow': flow[2][:, :2],
            'flow_tea': flow_teacher,
            'loss_l1': loss_l1,
            'loss_l1_extracted': loss_l1_extracted,
            'loss_l1_mask': loss_l1_mask,
            'loss_l1_mask_original': loss_l1_mask_original,
            'loss_l1_original': loss_l1_original,
            'loss_total': loss_total
            }
