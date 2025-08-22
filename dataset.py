import os
import cv2
import ast
import torch
import numpy as np
import random
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import CenterCrop

cv2.setNumThreads(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class VimeoDataset(Dataset):
    def __init__(self, dataset_name, batch_size=32):
        self.batch_size = batch_size
        self.dataset_name = dataset_name        
        
        self.data_root = '../data/interpolation_data'
        self.image_root = os.path.join(self.data_root, 'sequences')
        train_fn = os.path.join(self.data_root, 'tri_trainlist.txt')
        test_fn = os.path.join(self.data_root, 'tri_testlist.txt')
        with open(train_fn, 'r') as f:
            self.trainlist = f.read().splitlines()
        with open(test_fn, 'r') as f:
            self.testlist = f.read().splitlines()   
        self.load_data()

    def __len__(self):
        return len(self.meta_data)

    def load_data(self):
        cnt = int(len(self.trainlist) * 0.95)
        if self.dataset_name == 'train':
            self.meta_data = self.trainlist[:cnt]
        elif self.dataset_name == 'test':
            self.meta_data = self.testlist
        else:
            self.meta_data = self.trainlist[cnt:]
           
    def crop(self, img0, gt, original_gt, img1, h, w):
        ih, iw, _ = img0.shape
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        img0 = img0[x:x+h, y:y+w, :]
        img1 = img1[x:x+h, y:y+w, :]
        gt = gt[x:x+h, y:y+w, :]
        original_gt = original_gt[x:x+h, y:y+w*3, :]
        return img0, gt, original_gt, img1

    def interleave_tensors(self, t1, t2, t3):
        
        # Assuming t1, t2, t3 are of shape [H,W,C]
        H, W, C = t1.shape
        result = np.zeros((H, W*3, C), dtype=t1.dtype)
        result[:, 0::3, :] = t1
        result[:, 1::3, :] = t2
        result[:, 2::3, :] = t3
        return result
    
    def getimg(self, index):
        imgpath = os.path.join(self.image_root, self.meta_data[index])
        imgpaths = [imgpath + '/im1.png', imgpath + '/im2.png', imgpath + '/im3.png']

        # Load images
        img0 = cv2.imread(imgpaths[0])
        gt = cv2.imread(imgpaths[1])
        img1 = cv2.imread(imgpaths[2])
        timestep = 0.5

        return img0, gt, img1, timestep
    
        # RIFEm with Vimeo-Septuplet
        # imgpaths = [imgpath + '/im1.png', imgpath + '/im2.png', imgpath + '/im3.png', imgpath + '/im4.png', imgpath + '/im5.png', imgpath + '/im6.png', imgpath + '/im7.png']
        # ind = [0, 1, 2, 3, 4, 5, 6]
        # random.shuffle(ind)
        # ind = ind[:3]
        # ind.sort()
        # img0 = cv2.imread(imgpaths[ind[0]])
        # gt = cv2.imread(imgpaths[ind[1]])
        # img1 = cv2.imread(imgpaths[ind[2]])        
        # timestep = (ind[1] - ind[0]) * 1.0 / (ind[2] - ind[0] + 1e-6)
            
    def __getitem__(self, index):        
        img0, gt, img1, timestep = self.getimg(index)
        original_gt = self.interleave_tensors(img0, gt, img1)
        
        if self.dataset_name == 'train':
            img0, gt, original_gt, img1 = self.crop(img0, gt, original_gt, img1, 160, 160)
            
            if random.uniform(0, 1) < 0.5:
                img0 = img0[:, :, ::-1]
                img1 = img1[:, :, ::-1]
                gt = gt[:, :, ::-1]
                original_gt = original_gt[:, :, ::-1]
            if random.uniform(0, 1) < 0.5:
                img0 = img0[::-1]
                img1 = img1[::-1]
                gt = gt[::-1]
                original_gt = original_gt[::-1]
            if random.uniform(0, 1) < 0.5:
                img0 = img0[:, ::-1]
                img1 = img1[:, ::-1]
                gt = gt[:, ::-1]
                original_gt = original_gt[:, ::-1]
            if random.uniform(0, 1) < 0.5:
                tmp = img1
                img1 = img0
                img0 = tmp
                timestep = 1 - timestep
            # # random rotation
            # p = random.uniform(0, 1)
            # if p < 0.25:
            #     img0 = cv2.rotate(img0, cv2.ROTATE_90_CLOCKWISE)
            #     gt = cv2.rotate(gt, cv2.ROTATE_90_CLOCKWISE)
            #     img1 = cv2.rotate(img1, cv2.ROTATE_90_CLOCKWISE)
            # elif p < 0.5:
            #     img0 = cv2.rotate(img0, cv2.ROTATE_180)
            #     gt = cv2.rotate(gt, cv2.ROTATE_180)
            #     img1 = cv2.rotate(img1, cv2.ROTATE_180)
            # elif p < 0.75:
            #     img0 = cv2.rotate(img0, cv2.ROTATE_90_COUNTERCLOCKWISE)
            #     gt = cv2.rotate(gt, cv2.ROTATE_90_COUNTERCLOCKWISE)
            #     img1 = cv2.rotate(img1, cv2.ROTATE_90_COUNTERCLOCKWISE)
        img0 = torch.from_numpy(img0.copy()).permute(2, 0, 1)
        img1 = torch.from_numpy(img1.copy()).permute(2, 0, 1)
        gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)
        original_gt = torch.from_numpy(original_gt.copy()).permute(2, 0, 1)
        timestep = torch.tensor(timestep).reshape(1, 1, 1)
        return torch.cat((img0, img1, gt), 0), timestep, original_gt



class OCTDenoisingDataset(Dataset):
    def __init__(self, clean_dir, noisy_dir, crop_size=480):
        self.clean_dir = clean_dir
        self.noisy_dir = noisy_dir
        self.noisy_images = sorted([f for f in os.listdir(noisy_dir) if f.endswith('.png')])
        self.clean_images = {}
        
        # Group noisy images by their corresponding clean image
        for f in self.noisy_images:
            clean_id = f[:4]
            if clean_id not in self.clean_images:
                self.clean_images[clean_id] = f'{clean_id}.png'
        
        self.center_crop = CenterCrop(crop_size)

    def aug(self, img0, img1, h, w):
        ih, iw = img0.shape
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        img0 = img0[x:x+h, y:y+w]
        img1 = img1[x:x+h, y:y+w]
        return img0, img1
    def __len__(self):
        return len(self.noisy_images)

    def __getitem__(self, index):
        noisy_img_name = self.noisy_images[index]
        clean_id = noisy_img_name[:4]
        
        # Read noisy image
        noisy_img_path = os.path.join(self.noisy_dir, noisy_img_name)
        noisy_img = cv2.imread(noisy_img_path, cv2.IMREAD_GRAYSCALE)
        
        
        # Read corresponding clean image
        clean_img_name = self.clean_images[clean_id]
        clean_img_path = os.path.join(self.clean_dir, clean_img_name)
        clean_img = cv2.imread(clean_img_path, cv2.IMREAD_GRAYSCALE)
        

        clean_img, noisy_img = self.aug(clean_img, noisy_img, 480, 480)

        clean_img = torch.from_numpy(clean_img.copy()).float() / 255.
        noisy_img = torch.from_numpy(noisy_img.copy()).float() / 255.
        
        return clean_img.unsqueeze(0), noisy_img.unsqueeze(0)
    

class OCTDenoisingDatasetTrain(Dataset):
    def __init__(self, noisy_dir, crop_size=480):
        self.noisy_dir = noisy_dir
        self.noisy_images = sorted([f for f in os.listdir(noisy_dir) if f.endswith('.png') or f.endswith('.tif')])
        self.center_crop = CenterCrop(crop_size)

    def aug(self, img0, img1, h, w):
        ih, iw = img0.shape
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        img0 = img0[x:x+h, y:y+w]
        img1 = img1[x:x+h, y:y+w]
        return img0, img1
    def __len__(self):
        return len(self.noisy_images)

    def __getitem__(self, index):
        noisy_img_name = self.noisy_images[index]

        # Read noisy image
        noisy_img_path = os.path.join(self.noisy_dir, noisy_img_name)
        noisy_img = cv2.imread(noisy_img_path, cv2.IMREAD_GRAYSCALE)

        noisy_img, _ = self.aug(noisy_img, noisy_img, 480, 480)

        noisy_img = torch.from_numpy(noisy_img.copy()).float() / 255.
        
        return noisy_img.unsqueeze(0), noisy_img_name
    
class OCTDenoisingDatasetTest(Dataset):
    def __init__(self, clean_dir, noisy_dir, crop_size=480):
        self.clean_dir = clean_dir
        self.noisy_dir = noisy_dir
        self.noisy_images = sorted([f for f in os.listdir(noisy_dir) if f.endswith('.png')])
        self.clean_images = sorted([f for f in os.listdir(clean_dir) if f.endswith('.png')])
        
        self.center_crop = CenterCrop(crop_size)

    def aug(self, img0, img1, h, w):
        ih, iw = img0.shape
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        img0 = img0[x:x+h, y:y+w]
        img1 = img1[x:x+h, y:y+w]
        return img0, img1
    def __len__(self):
        return len(self.noisy_images)

    def __getitem__(self, index):
        noisy_img_name = self.noisy_images[index]
        
        # Read noisy image
        noisy_img_path = os.path.join(self.noisy_dir, noisy_img_name)
        noisy_img = cv2.imread(noisy_img_path, cv2.IMREAD_GRAYSCALE)
        
        
        # Read corresponding clean image
        clean_img_name = noisy_img_name
        clean_img_path = os.path.join(self.clean_dir, clean_img_name)
        clean_img = cv2.imread(clean_img_path, cv2.IMREAD_GRAYSCALE)
        

        clean_img, noisy_img = self.aug(clean_img, noisy_img, 480, 480)
        

        clean_img = torch.from_numpy(clean_img.copy()).float() / 255.
        noisy_img = torch.from_numpy(noisy_img.copy()).float() / 255.
        # add channel dimension
        clean_img = clean_img.unsqueeze(0)
        noisy_img = noisy_img.unsqueeze(0)
        # repeat the image to change from channel 1 to 3
        clean_img = clean_img.repeat(3, 1, 1)
        noisy_img = noisy_img.repeat(3, 1, 1)
        return clean_img, noisy_img
